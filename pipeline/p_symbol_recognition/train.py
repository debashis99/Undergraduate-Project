import numpy as np
import os, sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import cv2
from keras import backend as K


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.utils import to_categorical
import pickle

f = open("data_pickle", "rb")
(X_train, y_train, X_test, y_test, X_val, y_val )= pickle.load(f)

# encode the labels
le = preprocessing.LabelEncoder()

y_all = y_train + y_test + y_val
y_all = np.array(y_all)
print(y_all.shape)
y_all = np.unique(y_all)
print(y_all)


le.fit(y_all)
y_train = le.transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)
np.save('labelClasses.npy', le.classes_)

# loading image from int to float
X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
X_val = np.asarray(X_val).astype('float32')

#normalising image
X_train /= 255
X_test /= 255
X_val /= 255

X_train = X_train.reshape(X_train.shape[0], 45 , 45 , 1)
X_test = X_test.reshape(X_test.shape[0], 45 , 45 , 1)
X_val = X_val.reshape(X_val.shape[0], 45 , 45 , 1)

batch_size = 256
nb_epoch=20
nb_filters=32
nb_pool=2
nb_conv=3
nb_classes = len(le.classes_)


Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)
Y_val = to_categorical(y_val, nb_classes)



model = Sequential()

model.add(Conv2D(nb_filters, ( nb_conv,nb_conv) , padding ='valid', input_shape= (45,45,1)) )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))

#model.add(Conv2D(32, (nb_conv, nb_conv) ))
#model.add(ELU(alpha=1.0))
#model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))

model.add(Conv2D(64, (nb_conv, nb_conv) ))
model.add(ELU(alpha=1.0))
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))

#model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              #optimizer='adadelta',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=batch_size,epochs=nb_epoch,validation_data=(X_val, Y_val))

score = model.evaluate(X_test, Y_test, verbose=0)

#print('Test score:', score[0])
print('Test accuracy:', score[1])
print("Error: %.2f%%" % (100-score[1]*100))
model.save('my_model_withoutdrop.h5')
