from CLASSES import MulticlassLinearRegressionPredictor
from testFeatEstimation import full_test
from trainFeatEstimation import full_train

# uploading data:
testID = full_test['ID'].values
cols = full_train.columns.tolist()
new_cols = [cols[0], cols[1], cols[3], cols[5], cols[6], cols[4]]
X_test, X_train, y_train  = full_test[new_cols], full_train[new_cols], full_train[[cols[2]]].to_numpy().ravel()

# encoding categorical data:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ctEn = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ["location", "sex"])], remainder='passthrough')
X_train = ctEn.fit_transform(X_train)
X_test = ctEn.transform(X_test)
# encoding labels for training set:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# feature scaling:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 5:] = sc.fit_transform(X_train[:, 5:])
X_test[:, 5:] = sc.transform(X_test[:, 5:])

# training and evaluating model:
lambd = 0.1
from sklearn.metrics import confusion_matrix
LR = MulticlassLinearRegressionPredictor(X_train, y_train, labels_num=3, lambdaVal=lambd)
self_pred = LR.predict(X_train)
cm = confusion_matrix(y_train, self_pred)
print("Confusion  matrix:" + str(cm))

# predicting test:
prediction = LR.predict(X_test)
prediction = le.inverse_transform(prediction.astype(int))

from xlsxwriter import Workbook
workbook = Workbook('Results')
worksheet = workbook.add_worksheet()
row = 1
column = 0
worksheet.write(0, column, 'ID')
worksheet.write(0, column+1, 'species')
for item in prediction:
    worksheet.write(row, column, testID[row-1])
    worksheet.write(row, column+1, item)
    row += 1
workbook.close()
