import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from trainFeatEstimation import full_train, insertPredictions, location_incoding

# uploading data:
path = r"C:\Shir ASUS laptop\New projects\Bird classification\Dataset"
testSet = pd.read_csv(path + r"\test_set.csv")
species = full_train['species']
yBill = np.array(full_train['bill_length'])
yWing = np.array(full_train['wing_length'])
trainingSet = full_train.drop('species', axis=1)

# switching loc_ to inx:
testSet = location_incoding(testSet)

# Imputing bill_depth and mass features:
from sklearn.impute import SimpleImputer
imputer0 = SimpleImputer(strategy='mean')
imputer0.fit(trainingSet[["bill_depth", "mass"]])
testSet[["bill_depth", "mass"]] = imputer0.transform(testSet[["bill_depth", "mass"]])
# Imputing sex and location features:
imputer1 = SimpleImputer(strategy='most_frequent')
imputer1.fit_transform(trainingSet[["sex", "location"]])
testSet[["sex", "location"]] = imputer1.fit_transform(testSet[["sex", "location"]])

# removing wing and bill lengths from data:
ID = testSet['ID']
X_test = testSet.drop(labels=['bill_length', 'wing_length', 'ID'], axis=1)
X_train = trainingSet.drop(labels=['bill_length', 'wing_length'], axis=1)

# rearranging data:
cols = X_test.columns.tolist()
cols = [cols[1], cols[3], cols[0], cols[2]]
X_test = X_test[cols]

# encoding categorical data:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ctEn = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ["location", "sex"])], remainder='passthrough')
X_train = ctEn.fit_transform(X_train)
X_test = ctEn.fit_transform(X_test)

# feature scaling:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 5:] = sc.fit_transform(X_train[:, 5:])
X_test[:, 5:] = sc.transform(X_test[:, 5:])

# splitting data to relevant samples for the two featues:
XB_test = X_test[testSet[['bill_length']].isna().any(axis=1)]
XW_test = X_test[testSet[['wing_length']].isna().any(axis=1)]

# creating poly features for bill_length estimation:
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
XB_train_poly = np.hstack((X_train[:, :5], poly.fit_transform(X_train[:, 5:])))
XB_test_poly = np.hstack((XB_test[:, :5], poly.transform(XB_test[:, 5:])))

# evaluating model for bill estimation:
from CLASSES import LinearRegressionPredictor
iter = 500
alphaB = 0.35
LRB = LinearRegressionPredictor(XB_train_poly, yBill, iterations=iter, alpha=alphaB)
predB = LRB.predict(XB_test_poly, plot_J=False)

# evaluating model for wing estimation:
alphaW = 0.1
LRW = LinearRegressionPredictor(X_train, yWing, iterations=iter, alpha=alphaW)
predW = LRW.predict(XW_test, plot_J=False)


# inserting back predicted features:
insertPredictions(testSet, 'bill_length', predB)
full_test = insertPredictions(testSet, 'wing_length', predW)
