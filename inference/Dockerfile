FROM pytorchlightning/pytorch_lightning:latest

RUN apt-get update && apt-get install -y python3-opencv

WORKDIR /code/
COPY . /code/

RUN pip install -r requirements.txt