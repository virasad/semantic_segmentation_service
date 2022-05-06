import os

def mkdir_p(dirname: str):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)