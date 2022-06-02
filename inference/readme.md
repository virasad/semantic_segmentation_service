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
uvicorn main:app --reload
```
## See API doc
http://127.0.0.1:8000/docs

## Send Request
### Send request to predict image
```json
{
    "image": "base62 encoded image",
    "labeled_image" : "If True then return segmented image",
    "labelmap_path" : "Path to labelmap.txt", -required if labeled_image is True
}
```
### Python Example
```python
import requests
base64image = ''
url = 'http://localhost:8000/predict'
data = {'image': base64image}
r = requests.post(url, json=data)
print(r.json())

```
## Set Model
### Send request to set model
```json
{
  "model_path": "./path/to/model.pth"
}
```

### Python Example
```python
import requests
url = 'http://localhost:8000/set_model'
model_path = './model/model.pth'
data = {'model_path': model_path}
r = requests.post(url, json=data)
print(r.json())
```