FROM pytorchlightning/pytorch_lightning:latest
FROM tiangolo/uvicorn-gunicorn-fastapi:latest
RUN pip install lightning-flash
RUN pip install preimutils

WORKDIR /code/
COPY . /code/