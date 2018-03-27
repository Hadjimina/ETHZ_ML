import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC



# 1. get training and test set of X & Y
import csv
X_train = []
Y_train = []
X_test  = []


# Training datasets
with open('train_vale.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV, None)  # skip the header
    for row in readCSV:
        X_train.append(list( row[2:]))
        Y_train.append(float(row[1]))

with open('test_vale.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV, None)  # skip the header
    for row in readCSV:
        X_test.append(list(map(float, row[1:])))

# Test datasets
# 10-fold split of train_X
# RepeatedKFold ??

# kf = KFold(n_splits=10)
# for l in lamda:
#     erro_vec = []
#     for train, test in kf.split(X):
        #generate training and test sets using kfold gugus
        # X_array = np.array(X)
        # Y_array = np.array(Y)
        # X_train, , Y_train, Y_test = X_array[train], X_array[test], Y_array[train], Y_array[test]
        # # do ridge regressions with lambda = l[i]
print("potato")
clf = OneVsRestClassifier(LinearSVC())
clf.fit(X_train, Y_train)
        # Generate prediction using the ridgre regression prediction on our x_test set
Y_test_predict = clf.predict(X_test)
        # Calculate error of our prediction
        #acc = accuracy_score(Y_train, Y_test_predict)
        #print("acc "+str(acc))
    #     erro_vec.append(RMSE)
    #
    # error_matrix.append(erro_vec)

print("potato1")
f = open('output.csv', 'w')
q = 2000
for v in Y_test_predict:
    st = str(q)+", "+str(v)[:-2]+"\n"
    print(st)
    q+=1
    f.write(st)
f.close()
