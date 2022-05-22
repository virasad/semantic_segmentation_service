FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /code/
COPY requirements.txt /code/

RUN pip install -r requirements.txt
