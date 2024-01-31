import numpy as np
import os, sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import cv2
import pickle
import numpy as np
import sys, cv2, math, os, sys
                                                     
from keras.models import load_model
from skimage import img_as_ubyte
from skimage.io import imread
from skimage.filters import gaussian, threshold_minimum
from skimage.morphology import square, erosion, thin

from skimage.util import invert
from skimage.morphology import skeletonize
from skimage import img_as_ubyte



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def img_pre_proc(img, h, w):
    img = cv2.GaussianBlur(img, (3,3),0)
    
    n_block_size = 65
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,n_block_size,10)

    img = invert(img)
    img[img == 255] = 1
    img = skeletonize(img)
    img = invert(img) 
    img = img_as_ubyte(img)

        # squareing the image
    dif = abs(int(w-h))
    pad = int(dif/2)
    White = [255,255,255]
    if(w > h):
        img= cv2.copyMakeBorder(img, pad , dif - pad ,0,0,cv2.BORDER_CONSTANT,value=White)
    else:
        img= cv2.copyMakeBorder(img ,0,0,pad ,dif  - pad ,cv2.BORDER_CONSTANT,value=White)

    # # resize to (45, 45)
    # img = cv2.resize(img,(45, 45))
    img = image_resize(img, h, w)
    img = cv2.resize(img, (h, w))
    n_block_size = 65
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,n_block_size,10)

    return img

data_path = "./img_data"

all_symbols = os.listdir(data_path)

print(all_symbols)

X = []
y = []
symbol_count = {}

count = 0

for each_s in all_symbols:
    images_path = data_path + '/' + str(each_s)
    all_images_list = os.listdir(images_path)
    symbol_count[each_s] = 0
    for each_i in all_images_list:
        img_path = images_path + '/' + str(each_i)
        img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE )

        img = img_pre_proc(img, 45, 45)

        # cv2.imshow('image', img)
        # cv2.waitKey(0)  
        # cv2.destroyAllWindows()

        X = X + [img]
        y = y + [each_s]
        symbol_count[each_s] = symbol_count[each_s] + 1
        print("Reading symbol " + each_s + " :: current count = " + str(symbol_count[each_s]) )
        sys.stdout.write("\033[F") 
    
    print("Done reading symbol " + each_s + " :: count = " + str(symbol_count[each_s]))
 
 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=56)

f = open("data_pickle", "wb")
pickle.dump((X_train, y_train, X_test, y_test, X_val, y_val),f)
# -*- coding: utf-8 -*-

