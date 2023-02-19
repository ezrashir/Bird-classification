import numpy as np
import pandas as pd
import sklearn

# uploading data:
path = r"C:\Shir ASUS laptop\New projects\Bird classification\Dataset"
trainingSet, trainingTarget = pd.read_csv(path + r"\training_set.csv"), pd.read_csv(path + r"\training_target.csv")
full_train = pd.concat([trainingSet, trainingTarget["species"]], axis=1)

# checking the number of missing items for each feature:
BDnum, BLnum, WLnum, Lnum, Mnum, Snum = full_train["bill_depth"].isna().sum(), full_train["bill_length"].isna().sum(), full_train["wing_length"].isna().sum(), full_train["location"].isna().sum(), full_train["mass"].isna().sum(), full_train["sex"].isna().sum()

# switching loc_ to inx:
def location_incoding(dataSet):
    dataSet["location"] = dataSet["location"].replace(to_replace="loc_1", value=1)
    dataSet["location"] = dataSet["location"].replace(to_replace="loc_2", value=2)
    dataSet["location"] = dataSet["location"].replace(to_replace="loc_3", value=3)
    return dataSet
full_train = location_incoding(full_train)

# rearranging data:
cols = full_train.columns.tolist()
cols = [cols[6], cols[3], cols[5], cols[7], cols[0], cols[4], cols[1], cols[2]]
full_train = full_train[cols]

# Imputing bill_depth and mass features:
from sklearn.impute import SimpleImputer
imputer0 = SimpleImputer(strategy='mean')
full_train[["bill_depth", "mass"]] = imputer0.fit_transform(full_train[["bill_depth", "mass"]])
# Imputing sex and location features:
imputer1 = SimpleImputer(strategy='most_frequent')
full_train[["sex", "location"]] = imputer1.fit_transform(full_train[["sex", "location"]])

# removing data that needs to be predicted::
full_train = full_train.drop(labels=['ID'], axis=1)
trainData = full_train.dropna(axis=0, subset=["bill_length", "wing_length"])

# encoding categorical data:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ctEn = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ["location", "sex", "species"])], remainder='passthrough')
trainData = ctEn.fit_transform(trainData)

# Splitting data to train and test:
from sklearn.model_selection import train_test_split
features = trainData[:, 0:10]
X_train, X_val, yB_train, yB_val = train_test_split(features, trainData[:, 10], test_size=0.25, random_state=1)
_, _, yW_train, yW_val = train_test_split(features, trainData[:, 11], test_size=0.25, random_state=1)

# feature scaling:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 8:] = sc.fit_transform(X_train[:, 8:])
X_val[:, 8:] = sc.transform(X_val[:, 8:])

# creating poly features for bill_length estimation:
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_train_poly = np.hstack((X_train[:, :8], poly.fit_transform(X_train[:, 8:])))
X_val_poly = np.hstack((X_val[:, :8], poly.transform(X_val[:, 8:])))

# training and evaluating model for bill estimation:
from CLASSES import LinearRegressionPredictor
iter = 500
alphaB = 0.35
LRB = LinearRegressionPredictor(X_train_poly, yB_train, iterations=iter, alpha=alphaB)
predB = LRB.predict(X_val_poly, plot_J=False)
print("mean displacement = " + str(np.mean(yB_val-predB)))
tB = sklearn.metrics.mean_absolute_error(yB_val, predB)/np.mean(full_train['bill_length'])

# training and evaluating model for wing estimation:
alphaW = 0.1
LRW = LinearRegressionPredictor(X_train, yW_train, iterations=iter, alpha=alphaW)
predW = LRW.predict(X_val, plot_J=False)
print("mean displacement = " + str(np.mean(yW_val-predW)))
tW = sklearn.metrics.mean_absolute_error(yW_val, predW)/np.mean(full_train['wing_length'])

############## predicting unknown values of beak and wing lengths: ##############
X_test = full_train.drop(labels=['bill_length', 'wing_length'], axis=1)

# encoding categorical data:
ctEn1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ["location", "sex", "species"])], remainder='passthrough')
X_test = ctEn1.fit_transform(X_test)

# feature scaling:
X_test[:, 8:] = sc.transform(X_test[:, 8:])

# splitting data to relevant samples for the two featues:
XB_test = X_test[full_train[['bill_length']].isna().any(axis=1)]
XW_test = X_test[full_train[['wing_length']].isna().any(axis=1)]

# creating poly features for bill_length estimation:
XB_test_poly = np.hstack((XB_test[:, :8], poly.fit_transform(XB_test[:, 8:])))

# predicting missing features with the trained model:
bill_length = LRB.predict(XB_test_poly, plot_J=False)
wing_length = LRW.predict(XW_test, plot_J=False)


# inserting back predicted features:
def insertPredictions(dataframe, featureName, predictedVals):
    m=0
    for i in range(np.size(dataframe[featureName])):
        if pd.isna(dataframe.loc[i, featureName]):
            dataframe.loc[i, featureName] = predictedVals[m]
            m += 1
    return dataframe
full_train = insertPredictions(full_train, 'bill_length', bill_length)
full_train = insertPredictions(full_train, 'wing_length', wing_length)






