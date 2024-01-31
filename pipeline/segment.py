import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt



class Segment:
    def __init__(self,img, x, y, h, w):
        self.img = img
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.center = [ x + w//2 , y + h//2  ] 
        self.next = (None, None)
        self.sup = (None, None)
        self.sub = (None, None)
        self.ar=0
        self.back=(None,None)
        self.bsup=(None,None)
        self.bsub=(None,None)
        self.val=0


def segmentation(img):

    x_t = int( 500 / img.shape[0])
    if(x_t == 0):
        x_t = 1
    y_t  = x_t

    img = cv2.resize(img, None, fx = x_t, fy = y_t )

    img_original = np.copy(img) 
    img_labeled = img 
    # img_labeled = cv2.erode(img_labeled,None,iterations = 2)

    img = cv2.GaussianBlur(img, (3,3),0)

    n_block_size = 65
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,n_block_size,10)

    # # may be played with 
    # img = cv2.dilate(img,None,iterations = 1)
    # img = cv2.erode(img,None,iterations = 3)
    # img = cv2.dilate(img,None,iterations = 1)

    img = cv2.bitwise_not(img, img)
    img = cv2.Canny(img,100,200) 

    img, contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda cont: cv2.boundingRect(cont)[0])

    segments = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        x = x + 1
        y = y + 1
        w = w - 1
        h = h - 1 
        box = img_original[y:y+h,x:x+w]
        
        if( w * h > 200):
            segment  = Segment(box,x, y, h, w )
            segments = segments + [segment]

            cv2.rectangle(img_labeled,(x,y),(x+w,y+h),(0,0,120),1) 
            # cv2.circle(img_labeled,(x + w//2, y + h//2), 5 , (0,0, 120), -1)

 
    return_v = {'img': img_original, 'img_labeled' : img_labeled, 'segments': segments}

    return return_v
        



if __name__ == "__main__":
    img_addr = str(sys.argv[1])
    segs = segmentation(img_addr)
    cv2.imshow('image',segs["img_labeled"])
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    


