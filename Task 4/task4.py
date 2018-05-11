import pandas as pd
import numpy as np
import math
import scipy
from sklearn.model_selection import KFold
from scipy.sparse.csgraph import connected_components
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

np.seterr(divide='ignore', invalid='ignore')

#read files
train_labeled = np.asarray(pd.read_hdf("train_labeled.h5", "train"))
train_unlabeled = np.asarray(pd.read_hdf("train_unlabeled.h5", "train"))
test = np.asarray(pd.read_hdf("test.h5", "test"))

#seperate train_labeled into x and y
X_train_labeled = train_labeled[:,1:]
Y_train_labeled = train_labeled[:,0]

kf = KFold(n_splits=3)
erro_vec = []
for train_k, test_k in kf.split(X_train_labeled):

    #generate training and test sets using kfold
    X_train_k, X_test_k, Y_train_k, Y_test_k = X_train_labeled[train_k], X_train_labeled[test_k], Y_train_labeled[train_k], Y_train_labeled[test_k]

    #fill an array of size len(train_unlabeled) with -1 meaning this are unlabeled
    Y_train_unlabeled = np.empty([len(train_unlabeled)])
    Y_train_unlabeled = np.full_like(Y_train_unlabeled, -1)

    #generate an array of all x Vectors and one with all y values
    X_train = np.concatenate((X_train_k, train_unlabeled))
    Y_train = np.concatenate((Y_train_k, Y_train_unlabeled))

    #perform semi-supervised learning with either LabelSpreading or LabelPropagation
    #kernels tried: knn, rbf ->knn is better
    label_prop_model = LabelSpreading(kernel='knn', gamma=20, n_neighbors=7, alpha=0.2, max_iter=4000, tol=0.001, n_jobs=1)
    #label_prop_model = LabelPropagation(kernel='knn', gamma=20, n_neighbors=7, alpha=None, max_iter=1000000, tol=0.001, n_jobs=1)
    label_prop_model.fit(X_train, Y_train)
    Y_predict = label_prop_model.predict(X_test_k)

    #compute prediction error and add to erro_vec
    RMSE = mean_squared_error(Y_test_k, Y_predict)**0.5
    erro_vec.append(RMSE)

print(np.mean(erro_vec))

#print result file
f = open('output.csv', 'w')
f.write("Id,y\n")
q = 30000
for v in Y_predict:
    st = str(q)+","+str(math.floor(v))+"\n"
    q+=1
    f.write(st)
f.close()
