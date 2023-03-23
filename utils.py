import os

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
