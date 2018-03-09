import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer


# 1. get training and test set of X & Y
import csv
X_dummy = []
y = []
X =[]
X_test = []
X_train = []
y_test = []
y_train= []

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
    X.append(temp)


# kf = KFold(n_splits=10)
# erro_vec = []
# for train, test in kf.split(X):
#     #generate training and test sets using kfold gugus
#     X_array = np.array(X)
#     y_array = np.array(y)
#     X_train, X_test, y_train, y_test = X_array[train], X_array[test],y_array[train], y_array[test]
#     # do ridge regressions with lambda = l[i]
#     regr = linear_model.LinearRegression()
#     regr.fit(X_train, y_train)
#     # Generate prediction using the ridgre regression prediction on our x_test set
#     y_test_predict = regr.predict(X_test)
#     # Calculate error of our prediction
#     RMSE = mean_squared_error(y_test, y_test_predict)**0.5
#     erro_vec.append(RMSE)

#test on entire data
regr = linear_model.LinearRegression()
regr.fit(X, y)

print("Averages")
print(str(np.mean(erro_vec))+"\n")



f = open('output.csv', 'w')
for c in regr.coef_:
    f.write(str(c)+"\n")
f.close()
