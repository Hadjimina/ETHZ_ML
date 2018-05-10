import pandas as pd
import numpy as np
import math
import scipy
from scipy.sparse.csgraph import connected_components
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.utils import shuffle

#read files
train_labeled = np.asarray(pd.read_hdf("train_labeled.h5", "train"))
train_unlabeled = np.asarray(pd.read_hdf("train_unlabeled.h5", "train"))
test = np.asarray(pd.read_hdf("test.h5", "test"))

#seperate train_labeled into x and y
X_train_labeled = train_labeled[:,1:]
Y_train_labeled = train_labeled[:,0]

#fill an array of size len(train_unlabeled) with -1 meaning this are unlabeled
Y_train_unlabeled = np.empty([len(train_unlabeled)])
Y_train_unlabeled = np.full_like(Y_train_unlabeled, -1)

#generate an array of all x Vectors and one with all y values
X_train = np.concatenate((X_train_labeled, train_unlabeled))
Y_train = np.concatenate((Y_train_labeled, Y_train_unlabeled))

#Shuffle X_train and Y_train in same manner to get randomization
X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

#perform semi-supervised learning
label_prop_model = LabelSpreading( kernel='knn', alpha=0.8, max_iter=4000)
#label_prop_model = LabelPropagation()
label_prop_model.fit(X_train, Y_train)
Y_predict = label_prop_model.predict(test)

#print result file
f = open('output.csv', 'w')
f.write("Id,y\n")
q = 30000
for v in Y_predict:
    st = str(q)+","+str(math.floor(v))+"\n"
    q+=1
    f.write(st)
f.close()
