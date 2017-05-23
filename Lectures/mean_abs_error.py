# Load the dataset
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_linnerud
from sklearn.model_selection import learning_curve

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
seed = np.random.seed(5)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
reg1 = DecisionTreeRegressor()
reg1.fit(X_train, y_train)
print "Decision Tree mean absolute error: {:.2f}".format(mae(y_test, reg1.predict(X_test)))

reg2 = LinearRegression()
reg2.fit(X_train, y_train)
print "Linear regression mean absolute error: {:.2f}".format(mae(y_test, reg2.predict(X_test)))

learning_curve(LinearRegression(), X, y, cv=train_test_split(), train_sizes=20)