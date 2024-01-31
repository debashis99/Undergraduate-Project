import numpy as np
import sys, cv2, math, os, sys

from skimage import img_as_ubyte
from skimage.io import imread
from skimage.filters import gaussian, threshold_minimum
from skimage.morphology import square, erosion, thin

from skimage.util import invert
from skimage.morphology import skeletonize
from skimage import img_as_ubyte

from sklearn import preprocessing




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


def predict(X,model):
    script_dir = os.path.dirname(__file__)

    le = preprocessing.LabelEncoder()
    le.classes_ = np.load( os.path.join(script_dir, 'h_symbol_recognition/labelClasses.npy' ))

    X = np.asarray(X).astype('float32')
    X /= 255
    X = X.reshape(X.shape[0], 45 , 45 , 1)

    Y_pred = model.predict(X)
    y_pred = Y_pred.argmax(1)

    y_label = le.inverse_transform(y_pred)

    return y_label


def label_segments(model,segments):
    for segment in segments:
        img  = segment.img  
        h = segment.h
        w = segment.w 

        if(h <= 0 or w <= 0):
            # to do: delete this segment
            continue

        img_c = np.copy(img)
        n_block_size = 65
        img_c = cv2.adaptiveThreshold(img_c,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,n_block_size,10)

        sh = img.shape
        per_black= 1 - np.sum(img_c)/(255*img_c.size)

        img = img_pre_proc(img, 45, 45)
        p = predict([img], model)

        l = str(p[0])
        segment.label = str(p[0]).lower()
        
        # if(l == 'X'):
        #     segment.label = 'x'
        
        # if(l == 'A'):
        #     segment.label = 'a'
        
        segment.ar=(w/h)

    return segments
  


    