FROM pytorchlightning/pytorch_lightning:latest

RUN apt-get update && apt-get install -y python3-opencv
RUN apt-get install python3-tk

WORKDIR /code/
COPY . /code/

RUN pip install -r requirements.txt