from __future__ import absolute_import, print_function, division

import time
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, \
    ShuffleSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer
import pandas as pd
import numpy as np

le = LabelEncoder()
ohe = OneHotEncoder()

data = pd.read_csv('titanic_data.csv')
data = data.dropna().reset_index(drop=True)

labels = data.Survived
features = data.drop(['Survived'], axis=1)

features = pd.DataFrame(
    [pd.Series(le.fit_transform(features[column]), name=column)
     if features[column].dtype == 'object'
     else features[column] for column in features]).T

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels)

clf = SVC(max_iter=5000)
kernel = ["rbf", "sigmoid", "poly", "linear"]
C = np.linspace(0.1, 1, 20)
gamma = np.linspace(0.1, 1, 20)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
n_iter = 500
grid = RandomizedSearchCV(estimator=clf, param_distributions=dict(kernel=kernel,
                                                                  C=C,
                                                                  gamma=gamma),
                          scoring=make_scorer(accuracy_score), cv=cv,
                          n_iter=n_iter
                          )
grid = grid.fit(features, labels)
print(grid)
print(grid.best_score_, grid.best_params_)
