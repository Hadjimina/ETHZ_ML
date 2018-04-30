import tensorflow as tf
import pandas as pd
import numpy as np
import math
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
np.random.seed(123)  # for reproducibility
train = np.asarray(pd.read_hdf("train.h5", "train"))
test = np.asarray(pd.read_hdf("test.h5", "test"))

#train has dimenstions of 45324x100
X_train = train[:,1:]
Y_train = train[:,0]
Y_train = tf.keras.utils.to_categorical(Y_train)

print(test.shape)

model = Sequential()


#acc = 0.899
# model.add(Dense(512,activation='relu',input_shape=(100,)))
# model.add(Dense(512,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(5,activation='softmax'))
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])


model.add(Dense(512,activation='relu',input_shape=(100,)))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



model.fit(X_train, Y_train,
          batch_size=64, nb_epoch=150, verbose=1)

Y_predict = model.predict(test)
Y_predict = np.argmax(np.asarray(Y_predict), 1)

f = open('output.csv', 'w')
f.write("Id,y\n")
q = 45324
for v in Y_predict:

    st = str(q)+","+str(v)+"\n"
    #print(st)
    q+=1
    f.write(st)
f.close()
