import numpy as np
import sys, cv2, math, os, sys

from segment import segmentation
from label import label_segments
from layout import sanatize, label_neighbours, get_latex
from h_symbol_recognition import predict

from keras.models import load_model

script_dir = os.path.dirname(__file__)

model_path = os.path.join(script_dir, 'h_symbol_recognition/my_model.h5')


def img_to_latex(path = None, stream = None):
    if path is not None:
        image = cv2.imread(path,0)
    else:	
        data = stream.read()
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, 0)


    img_segment = segmentation(image)
    segments = img_segment["segments"]   
    img  = img_segment["img_labeled"]

    cv2.imshow('image', img )
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    
   
    model = load_model(model_path)
    segments = label_segments(model,segments)
    segments = sanatize(segments,img)
    segments = label_neighbours(segments)
    for seg in segments:
        if seg.next[0] != None:
            nexts = seg.next[0]
            cv2.line(img,(seg.center[0],seg.center[1]),(nexts.center[0],nexts.center[1]),(0,0, 255),5)
            cv2.circle(img,(seg.center[0],seg.center[1]), 7, (0,0,255), -1)
            cv2.circle(img,(nexts.center[0],nexts.center[1]), 7, (0,0,255), -1)
        if seg.sub[0] != None:
            nexts = seg.sub[0]
            cv2.circle(img,(seg.center[0],seg.center[1]), 7, (0,0,255), -1)
            cv2.circle(img,(nexts.center[0],nexts.center[1]), 7, (0,0,255), -1)
            cv2.line(img,(seg.center[0],seg.center[1]),(nexts.center[0],nexts.center[1]),(0,0, 255),2)
        if seg.sup[0] != None:
            nexts = seg.sup[0]
            cv2.circle(img,(seg.center[0],seg.center[1]), 7, (0,0,255), -1)
            cv2.circle(img,(nexts.center[0],nexts.center[1]), 7, (0,0,255), -1)
            cv2.line(img,(seg.center[0],seg.center[1]),(nexts.center[0],nexts.center[1]),(0,0, 255),1)

    cv2.imshow('image', img )
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    
    latex = '$' + get_latex(segments[0]) + '$ // //'
    print(latex)
    return (latex, img)

if __name__ == "__main__":
    img_addr = str(sys.argv[1])
    img_to_latex(path = img_addr)
