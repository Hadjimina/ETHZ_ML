import tensorflow as tf
import pandas as pd
import numpy as np
import math
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# for reproducibility
np.random.seed(123)

#read files
train_labeled = np.asarray(pd.read_hdf("train_labeled.h5", "train"))
train_unlabeled = np.asarray(pd.read_hdf("train_unlabeled.h5", "train"))
test = np.asarray(pd.read_hdf("test.h5", "test"))

#seperate x vector and y value
X_train_labeled = train_labeled[:,1:]
Y_train_labeled = train_labeled[:,0]
Y_train_labeled = tf.keras.utils.to_categorical(Y_train_labeled)
print(test.shape)

#Setup Neural Network sequential model
model = Sequential()

#Add different layers
model.add(Dense(512,activation='relu',input_shape=(128,)))
model.add(Dropout(0.10))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(10,activation='softmax'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

#Fit the labeled training data & predict the unlabeled data & take class with highest prob.
model.fit(X_train_labeled, Y_train_labeled, batch_size=64, nb_epoch=150, verbose=1)
Y_predict_unlabeled = model.predict(train_unlabeled)
Y_predict_unlabeled = np.argmax(np.asarray(Y_predict_unlabeled), axis=1)
Y_predict_unlabeled = tf.keras.utils.to_categorical(Y_predict_unlabeled, num_classes=10)

#Concatenate the labeled training set and the now predicted unlabeled training sets
X_train = np.concatenate((X_train_labeled, train_unlabeled))
Y_train = np.concatenate((Y_train_labeled, Y_predict_unlabeled))

#Fit the new X and Y vectors in the Neural Network
model.fit(X_train, Y_train, batch_size=64, nb_epoch=150, verbose=1)
Y_predict = model.predict(test)
Y_predict = np.argmax(np.asarray(Y_predict), 1)

#Print result file
f = open('output.csv', 'w')
f.write("Id,y\n")
q = 30000
for v in Y_predict:
    st = str(q)+","+str(math.floor(v))+"\n"
    q+=1
    f.write(st)
f.close()
