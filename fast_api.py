import os, random, time, cv2, io, warnings
from fastapi import FastAPI, File, UploadFile, Request, Response, HTTPException
# TODO: test out fastapi-health: https://github.com/Kludex/fastapi-health
app_version = '0.01'
app = FastAPI(
	title = 'Instafilter',
	description = 'instagram-style filters in python',
	version = app_version
)

# rate limiter
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
limiter = Limiter(key_func=get_remote_address, strategy = 'fixed-window-elastic-expiry',
			headers_enabled = True)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
RATE_LIMIT = os.getenv("RATE_LIMIT", '1/second;60/minute')

from PIL import Image
import numpy as np

from instafilter import Instafilter

#----------- API data type definition -----------#
from typing import List, Optional, Union
from pydantic import BaseModel

class ClipObj(BaseModel):
	start_frame: int
	end_frame: int
	keyframes: Optional[List[int]] = None

class MetadataObj(BaseModel):
	frame_width: int
	frame_height: int
	frame_count: int
	fps: float
	duration: float
	hash_method: str
	threshold: float
	min_frame_count: int
	max_frame_count: int
	keyframes_threshold: float
	max_keyframes: int
	max_keyframes_rate: float
	cross_dissolve_threshold: float
	compute_time: float
	system_stats: dict
	psFileId: Optional[str] = None

class InfoObj(BaseModel):
	message: str
	status: str
	rate_limit: str

class ClipsFound(BaseModel):
	clips_found: List[ClipObj]
	clips_count: int
	metadata: MetadataObj
#--------------Machine Diagnostic----------------#
import psutil
def GetMachineStats():
	stats = {
		'cpu_use_percent': psutil.cpu_percent(percpu=True),
		'virtual_memory_use_percent':  psutil.virtual_memory().percent
	}
	return stats
#------------------------------------------------#
def image_to_bytes_array(PIL_Image, format = None, quality = 75):
	'''
	Takes a PIL Image and convert it to Binary Bytes
	PIL_Image.tobytes() is not recommended (see: https://stackoverflow.com/a/58949303/14285096)

	Args:
		quality: compression quality. passed to Image.save() (see: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg)
	'''
	imgByteArr = io.BytesIO()
	# from the PIL doc:
	# If a file object was used instead of a filename, this parameter should always be used.
	img_format = format if format else PIL_Image.format
	img_format = img_format if img_format else 'jpeg'
	PIL_Image.save(imgByteArr, format = img_format, quality = quality)
	imgByteArr = imgByteArr.getvalue()
	return imgByteArr

def GetImage(image_url = None, file_obj = None, return_pil_im = False):
	'''
	Our General Purpose Image Loading Function
	Args:
		file_obj: a file object uploaded using the UploadFile method
	'''
	timestr = time.strftime("%Y%m%d-%H%M%S")

	if image_url:
		print(f'{timestr}: Downloading image from {image_url}...')
		pil_im = get_pil_im(image_url)
	elif file_obj:
		print(f'{timestr}: Downloading {file_obj.filename} {file_obj.content_type}...')
		# pil_im = Image.open(io.BytesIO(file_obj))
		file_obj.file.seek(0) # in case the file object was read before
		pil_im = Image.open(io.BytesIO(file_obj.file.read()))
	else:
		return None
	return pil_im if return_pil_im else np.array(pil_im)

# def upload_file_handler(file_obj, tmp_dir = '/tmp/miro/clipper/', overwrite = False):
# 	'''
# 	take a file object uploaded using UploadFile, save to disk and
# 	return the file path
# 	'''
# 	f_name = file_obj.filename
# 	fpath = os.path.join(tmp_dir, f_name)
#
# 	if os.path.isfile(fpath) and not overwrite:
# 		print(f'upload_file_handler: {fpath} already exists and will be reused.')
# 	else:
# 		if not os.path.isdir(tmp_dir):
# 			warnings.warn(f'upload_file_handler: {tmp_dir} does not exist. Making directory...')
# 			os.makedirs(tmp_dir)
#
# 		print(f'upload_file_handler: writing {file_obj.content_type} file object to {fpath}')
# 		with open(fpath, 'wb') as outfile:
# 			file_obj.file.seek(0) # in case the file object was read() before
# 			content = file_obj.file.read() # this is a bytes object
# 			outfile.write(content)
# 	return fpath
#
# @app.post("/detect", response_model = ClipsFound, response_model_exclude_unset = True)
# @limiter.limit(RATE_LIMIT)
# async def detect_by_video_upload(request: Request, response: Response,
# 					video_file: UploadFile = File(...), clip_det_threshold: float = 0.27,
# 					cross_dissolve_threshold: float = 0.19, hash_method: str = 'phash',
# 					keyframes_threshold: float = 0.1, max_keyframes: int = 0, max_keyframes_rate: float = 0.2,
# 					min_frame_count: int = 0, max_frame_count: int =0, lagger: int = 3,
# 					psFileId: str = ''):
# 	'''
# 	Clip Detection & Keyframe Extraction on Video File Upload
#
# 	* `video_file`: your uploaded video file
# 	* `hash_method`: phash (_default_), colorhash, whash, ahash, dhash
# 	* `clip_det_threshold`:
# 		% of hash difference from previous frame for the current frame to be considered a new clip
# 	* `cross_dissolve_threshold`:
# 		should be less than `clip_det_threshold`; 0 will turn off cross dissolve detection
# 	* `keyframes_threshold`:
# 		keyframe detection threshold, min % diff between neighboring frames to be considered peaks (i.e. keyframes)
# 	* `lagger`: how many frames to look back for cross dissolve detection
# 	* `max_keyframes`:
# 		maximum number of keyframes per clip, if None (_default_) given, `max_keyframes_rate` will be used;
# 		if given, `max_keyframes_rate` will be ignored.
# 	* `max_keyframes_rate`:
# 		set max number of keyframes per clip to % of number of total frames within the clip
# 	* `min_frame_count`: clips must have this number of frames minimum; if 0,
# 						it will be calculated as half of *FPS*
# 	* `max_frame_count`: just a placeholder, will not be used
# 	'''
# 	vid_path = upload_file_handler(file_obj = video_file, overwrite = True)
# 	return ck_detect(video_path = vid_path, clip_det_threshold = clip_det_threshold,
# 		cross_dissolve_threshold = cross_dissolve_threshold,
# 		hash_method = hash_method, keyframes_threshold = keyframes_threshold,
# 		max_keyframes = max_keyframes, max_keyframes_rate = max_keyframes_rate,
# 		min_frame_count= min_frame_count, max_frame_count = max_frame_count,
# 		lagger = lagger, psFileId = psFileId)

@app.post("/apply", response_class = Response,
			responses = {200: {"content": {"image/jpeg": {}}}})
@limiter.limit(RATE_LIMIT)
async def apply_instafilter(request: Request, response: Response,
	model_name: str, file_obj: UploadFile = File(...)):
	'''Apply Instafilter to an Image and returns it as a File Object
	'''
	im = GetImage(file_obj = file_obj)

	if model_name not in read_models():
		raise HTTPException(status_code = 404, detail = f"model: {model_name} not found.")

	model = Instafilter(model_name)
	im_out = model(im, is_RGB = True)
	pil_im_out = Image.fromarray(im_out)
	return Response(content = image_to_bytes_array(pil_im_out, quality = 100),
			media_type = "image/jpeg")

@app.get("/models", response_model = List[str])
def read_models():
	'''List available Filters'''
	return sorted(Instafilter.get_models())

@app.get("/")
def read_root():
	return {"Version": app_version,
			'status': 'Healthy',
			'message': 'see docs/ endpoint for help',
			}
