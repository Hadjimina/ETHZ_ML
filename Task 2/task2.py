import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score



# 1. get training and test set of X & Y
import csv
print("Hello World")
X = []
Y = []
X_test = []
accVec = []
avgVec = []
numFolds = 1
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
for d in degreeVec:
#     print("FOR DEGREE="+str(d))
#     kf = KFold(n_splits=numFolds)
#     w = 1
#     for train, test in kf.split(X):
#         print("iteration "+str(w)+"/"+str(numFolds)+"...")
#         w+=1
        #generate training and test sets using kfold gugus
        #X_array = np.array(X)
        #Y_array = np.array(Y)
        #X_train, X_test, Y_train, Y_test = X_array[train], X_array[test], Y_array[train], Y_array[test]
    X_train = X
    Y_train = Y
    # clf = OneVsRestClassifier(LinearSVC()) ---------- results in acc of 0.6
    clf = svm.SVC(kernel="poly", degree=,cache_size=2000) #linear,poly,rbf,sigmoid, precomputed  kernel='poly',degree=3
    clf.fit(X_train, Y_train)
    Y_test_predict = clf.predict(X_test)
    print(Y_test_predict)
    f = open('output_degree_'+ str(d)+'.csv', 'w')
    f.write("Id,y\n")
    q = 2000
    for v in Y_test_predict:
        f.wirte("Id,y\n")
        st = str(q)+","+str(v)[:-2]+"\n"
        #print(st)
        q+=1
        f.write(st)
    f.close()
        # acc = accuracy_score(Y_test, Y_test_predict)
        # accVec.append(acc)
        # if(acc <= 0.65):
        #     print("accuracy is SHIT "+str(acc))
        # else:
        #     print("accuracy is GOOD "+str(acc))
    # avg = np.mean(np.array(accVec))
    # if (avg < 0.65):
    #     print("AVERAGE accuracy is SHIT "+str(avg))
    # else:
    #     print("AVERAGE accuracy is GOOD "+str(avg))
    # avgVec.append(avg)




f = open('output.csv', 'w')
f.write("Id,y\n")
q = 2000
for v in Y_test_predict:
    f.write("Id,y\n")
    st = str(q)+","+str(v)[:-2]+"\n"
    #print(st)
    q+=1
    f.write(st)
f.close()
