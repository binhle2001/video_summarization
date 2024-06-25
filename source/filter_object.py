import os
import shutil
from .settings import *
def clean_object(image_path = OBJECT_IMAGE_PATH, coor_path = OBJECT_COORDINATE_PATH):
    for object in os.listdir(image_path):
        if len(os.listdir(image_path + '/' + object)) < 10:
            shutil.rmtree(image_path + '/' + object, ignore_errors=True)
            shutil.rmtree(coor_path + '/' + object, ignore_errors=True)