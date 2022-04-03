FROM pytorchlightning/pytorch_lightning:latest
RUN pip install lightning-flash[image]
RUN pip install opencv-python
RUN pip install fastapi
RUN pip install uvicorn

WORKDIR /code/
COPY . /code/
