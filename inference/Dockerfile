FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /code/
COPY requirements.txt /code/

RUN pip install -r requirements.txt
