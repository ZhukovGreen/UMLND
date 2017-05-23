import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X = pd.read_csv('titanic_data.csv')
X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y)



clf2 = GaussianNB()
clf2.fit(X_train, y_train)

cv = None
n_jobs = 5

print learning_curve(GaussianNB(), X, y, cv=cv, n_jobs=n_jobs)