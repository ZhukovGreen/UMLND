import pandas as pd
import numpy as np
import visuals as vs
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn.metrics import r2_score

data = pd.read_csv('housing.csv')
prices = data.MEDV
features = data.drop('MEDV', axis=1)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, prices,
                                                                     test_size=0.2,
                                                                     random_state=0)


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_predict, y_true)

    # Return the score
    return score


# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

from sklearn.metrics import make_scorer
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = dict(max_depth=np.arange(1, 11))

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    print grid.best_score_
    return grid.best_estimator_



regressor = DecisionTreeRegressor(criterion='mse', max_depth=5, max_features=None,
                                  max_leaf_nodes=None, min_impurity_split=1e-07,
                                  min_samples_leaf=1, min_samples_split=2,
                                  min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                                  splitter='best')

reg = fit_model(features, prices)
reg1 = regressor.fit(X_train, y_train)
print r2_score(y_test, regressor.predict(X_test))
