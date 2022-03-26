import requests

def set_model(url, dir):
    data = {'modelPath': dir}
    r = requests.post(url, data=data)
    return r.text


def predict_image(url, imagePath):
    data = {'imagePath': imagePath}
    r = requests.post(url, data=data)
    return r.text

if __name__ == '__main__':
    url = 'http://127.0.0.1:8000'
    model_dir = 'model_dir'
    imagePath = 'test.jpg'
    set_model(url=url + '/set_model', dir=model_dir)
    predict_image(url=url + '/predict', imagePath=imagePath)
