FROM python:3.7.8-slim

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# for opencv
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# for cython
RUN pip install cython

# for mxnet
RUN apt-get update && \
    apt-get install -y wget python-dev gcc && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# copy project
COPY . .
