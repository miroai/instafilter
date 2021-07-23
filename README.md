![instafilter](development/header_image.jpg)

[![PyVersion](https://img.shields.io/pypi/pyversions/instafilter.svg)](https://img.shields.io/pypi/pyversions/instafilter.svg)
[![PyPI](https://img.shields.io/pypi/v/instafilter.svg)](https://pypi.python.org/pypi/instafilter)

Modifiy images using Instagram-like filters in python. [Read how it works on Medium](https://medium.com/@travis.hoppe/instagram-filters-in-python-acc1ee7e67bc)!

    pip install git+https://github.com/miroai/instafilter.git@dev

This branch as a few fixes from the [original](https://github.com/thoppe/instafilter), namely:
* able to run **without** CUDA
* more flexible training script
* update streamlit app where transformed image can be downloaded at full quality

Example:

``` python
from instafilter import Instafilter

model = Instafilter("Lo-fi")
new_image = model("myimage.jpg")

# To save the image, use cv2
import cv2
cv2.imwrite("modified_image.jpg", new_image)
```

## Sample images

Browse samples for every filter in [`development/examples`](development/examples).

## Requirements

Main requirements are `opencv`, `torch==1.7.1` and `streamlit` (optional for demo app only) and can all be installed via `pip install -r development/requirements.txt`

## Train
To train a filter explicitly:
```bash
cd development/train_new_model/
python train.py --input path/to/original.jpg --target path/to/styled.jpg --model_name output_model_name
```

or passively, the following command will look for transformed images in `development/train_new_model/input/` that do not yet have a similarly named model in `instafilter/models/` and train new models for those transformed images found:
```bash
python train.py
```

On a [`g4dn.xlarge`](https://aws.amazon.com/ec2/instance-types/g4/) one filter takes between 90 minutes to 2 hours to train.

See the code in [`development/train_new_model`](development/train_new_model) or run `python development/train_new_model/train.py -h` for the full list of options.

## Credits

+ Made with ❤️ by [Travis Hoppe](https://twitter.com/metasemantic?lang=en).

+ Header image sourced from [Will Terra](https://unsplash.com/photos/qIY9mUKT540) and modified with instafilter.
