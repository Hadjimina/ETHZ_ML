import numpy as np
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold

# 1. get training and test set of X & Y
import csv
X = []
Y = []
X_test = []
accVec = []
avgVec = []
numFolds = 10
degreeVec = [3,4,5,6,7,8,9,10]
# Training datasets
with open('train_vale.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV, None)  # skip the header
    for row in readCSV:
        X.append(list(map(float, row[2:])))
        Y.append(float(row[1]))
csvfile.close()

with open('test_vale.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV, None)  # skip the header
    for row in readCSV:
        X_test.append(list(map(float, row[1:])))
csvfile.close()
print("files loaded")

# simple variance based feature selection
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
Xselected = sel.fit_transform(X)
Xselected_test = sel.fit_transform(X_test)

# do SVM with degree d
clf = svm.SVC(kernel="poly", degree=3,cache_size=200) #linear,poly,rbf,sigmoid, precomputed  kernel='poly',degree=3
clf.fit(Xselected, Y)
Y_test_predict = clf.predict(Xselected_test)

f = open('output_degree' + '.csv', 'w')
f.write("Id,y\n")
q = 2000
for v in Y_test_predict:
    st = str(q)+","+str(int(round(v)))+"\n"
    #print(st)
    q+=1
    f.write(st)
f.close()

