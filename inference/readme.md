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

## Send Request
```python
import requests
image_path = 'test.jpg'
url = 'http://localhost:8000/predict'
data = {'image': image_path}
r = requests.post(url, data=data)
print(r.json())

```
## Set Model
```python
import requests
url = 'http://localhost:8000/set_model'
modelPath = './model/model.pth'
data = {'modelPath': modelPath}
r = requests.post(url, data=data)
print(r.json())
```
## Run from command line
```bash
python main.py
```