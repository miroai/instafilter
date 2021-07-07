import os, random, time, cv2, io, warnings
from fastapi import FastAPI, File, UploadFile, Request, Response, HTTPException
# TODO: test out fastapi-health: https://github.com/Kludex/fastapi-health
app_version = 0.01
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
	# convert to BGR before and after, model's internal conversion is buggy
	im_out = model(im[:,:,::-1], is_RGB = False)[:,:,::-1]
	pil_im_out = Image.fromarray(im_out)

	# see how to return an image in FastAPI: https://stackoverflow.com/a/67497103/14285096
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
