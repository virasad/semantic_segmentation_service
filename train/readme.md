# Installation
First of all please install the dependencies:
- [pytorch](https://pytorch.org/get-started/locally/)
- [Flash images](https://lightning-flash.readthedocs.io/en/latest/installation.html)
- Flash:

`pip install lightning-flash`
## Addition for webservice

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

```bash
pip install 'lightning-flash[image]'
```
- other dependencies:
```bash
pip install -r requirements.txt 
```

# How to run?
## run from script

```bash
cd backbone
uvicorn main:app --reload
```
## See API doc
http://127.0.0.1:8000/docs


## response url
set `RESPONSE_URL` in environment variables to your response url
