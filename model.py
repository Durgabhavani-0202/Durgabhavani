# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 09:35:05 2022

@author: user
"""
!pip install numpy
!pip install imblearn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('Churn_Modelling.csv',decimal=',')

data.head()
data.shape
data.describe(include='all')
data.info()
data.isnull().sum()

data = data.drop(['RowNumber','CustomerId','Surname','Geography'],axis = 1)

for col in data.columns.values:
    if data[col].dtype == 'int64':
        continue
    else:
        data[col] = pd.get_dummies(data[col], drop_first = True)
        
plt.figure(figsize = (15,7))
sns.heatmap(data.corr(), annot=True,fmt='.2g')

sns.countplot(x='Exited', palette="Set3", data=data)

#### Balancing the dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

x = data.drop(['Exited'], axis=1)
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True, stratify=y)

param_grid = {'max_depth': [3, 5, 6],'max_features': [2, 4, 6],'n_estimators':[50, 100],'min_samples_split': [3, 5, 7]}
random_forest = RandomForestClassifier()
random_forest_grid = GridSearchCV(random_forest, param_grid, cv=5, refit=True, verbose=3, n_jobs=-2)
random_forest_grid.fit(x, y)
####print_best_model(random_forest_grid)
print(random_forest_grid.best_params_)
print(random_forest_grid.best_score_)

best_rf_estimator = RandomForestClassifier(max_depth=5, 
                                           max_features=6, 
                                           min_samples_split=7, 
                                           n_estimators=100)

best_rf_estimator.fit(X_train, y_train)

rf_predict_test = best_rf_estimator.predict(X_test)
accuracy_score(y_test, rf_predict_test)
print(classification_report(y_test, rf_predict_test))

from imblearn.over_sampling import SMOTE

columns = data.columns.tolist()
columns = [c for c in columns if c not in ['Exited']]
target = 'Exited'
state = np.random.RandomState(42)
X = data[columns]
Y = data[target]
print(X.shape)
print(Y.shape)

churn = data[data['Exited']==1]
stay = data[data['Exited']==0]

oversample = SMOTE()
X_res, Y_res = oversample.fit_resample(X, Y)

X.shape, Y.shape
X_res.shape, Y_res.shape

from collections import Counter
print('Original dataset shape{}'.format(Counter(Y)))
print('reshaped dataset shape{}'.format(Counter(Y_res)))

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_res, Y_res, test_size=0.20, shuffle=True, stratify=Y_res)


param_grid1 = {'max_depth': [3, 5, 6],'max_features': [2, 4, 6],'n_estimators':[50, 100],'min_samples_split': [3, 5, 7]}
random_forest1 = RandomForestClassifier()
random_forest_grid1 = GridSearchCV(random_forest, param_grid, cv=5, refit=True, verbose=3, n_jobs=-2)
random_forest_grid1.fit(X_res, Y_res)
####print_best_model(random_forest_grid)
print(random_forest_grid1.best_params_)
print(random_forest_grid1.best_score_)

best_rf_estimator1 = RandomForestClassifier(max_depth=5, 
                                           max_features=6, 
                                           min_samples_split=7, 
                                           n_estimators=100)

best_rf_estimator1.fit(X_train1, y_train1)

pickle.dump(best_rf_estimator1,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[608,1,43,2,77000,2,1,1,120000]]))