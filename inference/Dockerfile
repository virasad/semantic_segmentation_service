FROM pytorchlightning/pytorch_lightning:latest
FROM tiangolo/uvicorn-gunicorn-fastapi:latest
RUN pip install lightning-flash[image]

WORKDIR /code/
COPY . /code/