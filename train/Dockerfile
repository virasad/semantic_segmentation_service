FROM pytorchlightning/pytorch_lightning:latest
RUN pip install lightning-flash[image]
RUN pip install opencv-python
RUN pip install uvicorn
RUN pip install fastapi

WORKDIR /code/
COPY . /code/
