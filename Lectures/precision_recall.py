# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import pandas as pd
from sklearn import cross_validation
import numpy as np

# Load the dataset

X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
seed = np.random.seed(5)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state=seed)
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(recall(y_test, clf1.predict(X_test)),
                                                                  precision(y_test, clf1.predict(X_test)))

clf2 = GaussianNB()
clf2.fit(X_train, y_train)
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(recall(y_test, clf2.predict(X_test)),
                                                                  precision(y_test, clf2.predict(X_test)))

results = {
    "Naive Bayes Recall": 0.41,
    "Naive Bayes Precision": 0.71,
    "Decision Tree Recall": 0.48,
    "Decision Tree Precision": 0.51
}
