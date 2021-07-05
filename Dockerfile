ARG PY_VERSION=3.9
# Using base image provided by nginx unit
FROM nginx/unit:1.22.0-python${PY_VERSION}
# Alternatively you can use different tags from https://hub.docker.com/r/nginx/unit
LABEL maintainer="ohjho <john@miro.io>"

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir fastapi==0.58.0 python-multipart==0.0.5 slowapi==0.1.4 psutil==5.8.0

# OpenCV for python 3.9 requires special handling
COPY development/requirements.txt /apps/instafilter/requirements.txt
RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir opencv-python==4.5.2.54
RUN pip install --no-cache-dir $(grep -v '^ *#\|^opencv\|^numpy\|^streamlit' /apps/instafilter/requirements.txt | grep .)

COPY . /apps/instafilter
COPY fast_api.py /apps/instafilter
COPY instafilter/ /apps/instafilter
COPY nginx_config.json /docker-entrypoint.d/config.json
