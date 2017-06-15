from sklearn.tree import DecisionTreeClassifier


def classify(features_train, labels_train):
    ### your code goes here--should return a trained decision tree classifer
    clf = DecisionTreeClassifier()
    clf = clf.fit(features_train, labels_train)
    return clf


import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

#################################################################################


########################## DECISION TREE #################################



#### your code goes here
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

### be sure to compute the accuracy on the test set
acc = accuracy_score(labels_test, clf.predict(features_test))


def submitAccuracies():
    return {"acc": round(acc, 3)}
