import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
import sys


# 1. get training and test set of X & Y
import csv
X_dummy = []
y = []
X = []
X_test = []
X_train = []
y_test = []
y_train= []
coef = []
#best lambda is 0.1 for public score, for local score l = 0.074 is better but we get overfitting (i.e. worse public score)
lamda = [0.68,1,10,100,1000]
error_matrix = []

transforms = [ lambda x: x, lambda x: x**2,  np.exp,np.cos] #??

# Training datasets
with open('train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV, None)  # skip the header
    for row in readCSV:
        X_dummy.append(list(map(float, row[2:])))
        y.append(float(row[1]))

for v in X_dummy:
    temp = []
    for f in transforms:
        temp.extend(map(f,v))
    temp.append(1)
    #print(temp[0],temp[5],temp[10],temp[15],temp[20])
    X.append(temp)

min_error = 6000000
min_error_index = -1
index = 0
kf = KFold(n_splits=10)
for l in lamda:
    erro_vec = []
    for train, test in kf.split(X):
        #generate training and test sets using kfold gugus
        X_array = np.array(X)
        y_array = np.array(y)
        X_train, X_test, y_train, y_test = X_array[train], X_array[test], y_array[train], y_array[test]
        # do ridge regressions with lambda = l[i]
        clf = linear_model.Lasso(alpha=l)
        clf.fit(X_train, y_train)
        coef.append(clf.coef_)
        # Generate prediction using the ridgre regression prediction on our x_test set
        y_test_predict = clf.predict(X_test)
        # Calculate error of our prediction
        RMSE = mean_squared_error(y_test, y_test_predict)**0.5
        erro_vec.append(RMSE)
    error = np.mean(np.array(erro_vec))
    if(error<min_error):
        min_error = error
        min_error_index = index
    index = index+1
    print("Current error "+str(error)+" Current index "+str(index)+", Min error "+str(min_error)+", Min error index "+str(min_error_index)+" lambda "+str(l) )

print("MIN ERROR "+str(min_error)+" min lambda "+str(lamda[min_error_index]))
f = open('output.csv', 'w')
for c in coef[index]:
    f.write(str(c)+"\n")
f.close()
