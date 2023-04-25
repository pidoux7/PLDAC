import os
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import math as math

def load_images(path, gray_scale = False):
    """
    On charge les images et on le mets toutes avec les mêmes dimensions
    """
    images = []
    names = os.listdir(path)
    for f in os.listdir(path):
        img = None
        if gray_scale : 
            img = cv2.imread(os.path.join(path, f), 0)
        else :
            img = cv2.imread(os.path.join(path, f))
        if img is not None:
            img = cv2.resize(img, (1024,1024))
            images.append(img)
    s = sorted(zip(names,images))
    tuples = zip(*s)
    names , images = [list(tuple) for tuple in tuples]
    return images, names

def create_labels(i_names):
    """
    créer les labels des images négatives correspondant à un fichier txt vide avec le même nom que l'image
    """
    for i in range(len(i_names)):
        file = open("./Labels/img_neg/"+i_names[i][:-3]+"txt", "w")
        file.close()

path_images = './Images/img_neg/'
images,i_names = load_images(path_images, True)
create_labels(i_names)
