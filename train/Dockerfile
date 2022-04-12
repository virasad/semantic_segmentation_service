FROM pytorchlightning/pytorch_lightning:latest
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /code/
COPY . /code/
RUN pip install -r requirements.txt
