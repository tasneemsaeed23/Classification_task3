# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 21:29:26 2022

@author: Tasneem said
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
x = dataset.iloc[:,0:7]
y = dataset.iloc[:,8:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###Decision Tree Model:###
#finding the the mean absolute error:
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
Regressor = DecisionTreeRegressor()
Regressor.fit(X_train, y_train)
X_validation = Regressor.predict(X_test)
decision_tree_error = mean_absolute_error( y_test ,X_validation)
print(decision_tree_error)

###Random Forest Model:###
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

###KNN Model:###
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from scipy.sparse.sputils import matrix

#feature scaling 
X_scale = StandardScaler()
X_train = X_scale.fit_transform(X_train)
X_test = X_scale.transform(X_test)

KNN_mod = KNeighborsClassifier(n_neighbors= 27, p=2, metric='euclidean')
KNN_mod.fit(X_train, y_train.values.ravel())
X_valid_KNN = KNN_mod.predict(X_test)

cm = confusion_matrix(y_test, X_valid_KNN)
print(cm)

###Loggistic Regression:###
from sklearn.linear_model import LogisticRegression
logistic_Regression = LogisticRegression(max_iter= 180)
logistic_Regression.fit(X_train, y_train.values.ravel())
X_valid_logistic = logistic_Regression.predict(X_test)
logistic_error = mean_absolute_error(y_test, X_valid_logistic)
print(logistic_error)

###Comparing Results:###
print("Decision Tree Model :")
print("\n Accuracy: " + str(accuracy_score(y_test, X_validation) *100) + "%")
print("\n F Score : " + str(f1_score(y_test, X_validation) *100) + "%")

print("KNN Model :")
print("\n Accuracy: " + str(accuracy_score(y_test, X_valid_KNN) *100) + "%")
print("\n F Score : " + str(f1_score(y_test, X_valid_KNN) *100) + "%")

print("loggistic Regression Model :")
print("\n Accuracy: " + str(accuracy_score(y_test, X_valid_logistic) *100) + "%")
print("\n F Score : " + str(f1_score(y_test, X_valid_logistic) *100) + "%")
