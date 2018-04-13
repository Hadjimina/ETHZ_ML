import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


# 1. get training and test set of X & Y
import csv
X = []
Y = []
error_matrix = []
res = []
lamda = [0.1,1,10,100,1000]

# Training datasets
with open('train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV, None)  # skip the header
    for row in readCSV:
        X.append(list(map(float, row[2:])))
        Y.append(float(row[1]))

# Test datasets
# 10-fold split of train_X
# RepeatedKFold ??

kf = KFold(n_splits=10)
for l in lamda:
    erro_vec = []
    for train, test in kf.split(X):
        #generate training and test sets using kfold gugus
        X_array = np.array(X)
        Y_array = np.array(Y)
        X_train, X_test, Y_train, Y_test = X_array[train], X_array[test], Y_array[train], Y_array[test]
        # do ridge regressions with lambda = l[i]
        clf = Ridge(alpha=l)
        clf.fit(X_train, Y_train)
        # Generate prediction using the ridgre regression prediction on our x_test set
        Y_test_predict = clf.predict(X_test)
        # Calculate error of our prediction
        RMSE = mean_squared_error(Y_test, Y_test_predict)**0.5
        erro_vec.extend(RMSE)
        print("lamda "+str(l)+"Error "+str(RMSE))
    error_matrix.append(erro_vec)

f = open('output.csv', 'w')
for v in error_matrix:
    print(np.mean(v))
    f.write(str(np.mean(v))+"\n")
f.close()
