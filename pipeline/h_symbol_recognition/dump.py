import numpy as np
import os, sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import cv2
import pickle


data_path = "./extracted_images"

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
        X = X + [img]
        y = y + [each_s]
        symbol_count[each_s] = symbol_count[each_s] + 1
        print("Reading symbol " + each_s + " :: current count = " + str(symbol_count[each_s]) )
        sys.stdout.write("\033[F") 
    
    print("Done reading symbol " + each_s + " :: count = " + str(symbol_count[each_s]))
 
 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=56)

f = open("MathSymbols_train_test", "wb")
pickle.dump((X_train, y_train, X_test, y_test, X_val, y_val),f)
# -*- coding: utf-8 -*-

