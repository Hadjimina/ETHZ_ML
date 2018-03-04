
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# 1. get training and test set of X & Y
import csv
train_X = []
train_Y = []
test_X = []
test_Y = []

# Training datasets
with open('train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV, None)  # skip the header
    for row in readCSV:
        train_X.append(list(map(float, row[2:])))
        train_Y.append(float(row[1]))

# Test datasets
with open('test.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV, None)  # skip the header
    for row in readCSV:
        temp = list(map(float, row[1:]))
        test_X.append(temp)
        test_Y.append(sum(temp) / len(temp))


# 2. use linear_model.LinearRegression() from sklearn to get regression  (should data be normalized?)
regr = linear_model.LinearRegression()
# 3. train regression on training X & Y
regr.fit(train_X, train_Y)
# 4. use regression.predict(X test) to get prediction using test set
test_Y_predict = regr.predict(test_X)

print("Prediction (index 10):",
      test_Y_predict[10], "\nGround truth", test_Y[10])
# 5. print RMSE and other stuff
RMSE = mean_squared_error(test_Y, test_Y_predict)**0.5
# # The coefficients
print("\nRMSE: ", RMSE)
print('\nCoefficients: \n', regr.coef_)
# # Explained variance score: 1 is perfect prediction
print('\nVariance score: %.2f' % r2_score(test_Y, test_Y_predict))

# 6. Write solution to file
f = open('output2.csv', 'w')
i = 999
f.write("Id,y\n")
for x in test_Y:
    i += 1
    f.write(str(i) + "," + str(x) + "\n")
f.close()
