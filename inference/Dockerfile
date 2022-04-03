FROM pytorchlightning/pytorch_lightning:latest

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements.txt

WORKDIR /code/
COPY . /code/