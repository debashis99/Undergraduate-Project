import numpy as np
import os, sys, pickle, cv2

from keras.models import load_model
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from skimage.morphology import skeletonize, thin

script_dir = os.path.dirname(__file__)

model_path = os.path.join(script_dir, 'my_model.h5')


def predict_x_train(model):
	data_path = os.path.join(script_dir, "MathSymbols_train_test" )
	f = open(data_path, "rb")
	(X_train, y_train, X_test, y_test, X_val, y_val )= pickle.load(f)

	# encode the labels
	le = preprocessing.LabelEncoder()
	le.fit(y_train)
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

	Y_pred = model.predict(X_test)
	y_pred = Y_pred.argmax(1)


	cm = confusion_matrix(y_test, y_pred)

	per = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])

	print(per)

def predict(X,model):
	le = preprocessing.LabelEncoder()
	le.classes_ = np.load( os.path.join(script_dir, 'labelClasses.npy' ))

	X = np.asarray(X).astype('float32')
	X /= 255
	X = X.reshape(X.shape[0], 45 , 45 , 1)
	
	Y_pred = model.predict(X)
	y_pred = Y_pred.argmax(1)

	y_label = le.inverse_transform(y_pred)

	return y_label


if __name__ == "__main__":
	model = load_model(model_path)

