# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
import matplotlib.pyplot as plt

# Import supplementary visualizations code visuals.py
import visuals as vs

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)

################################################
# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)
#####################################################

fig = plt.figure()
for plt_num, feature in enumerate(features):
    graphs = fig.add_subplot(2, 2, plt_num + 1)
    graphs.scatter(features[feature], prices)
    lin_reg = np.poly1d(np.polyfit(features[feature], prices, deg=1))
    lin_x = np.linspace(features[feature].min(), features[feature].max(), 2)
    lin_y = lin_reg(lin_x)
    graphs.plot(lin_x, lin_y, color='r')
    graphs.set_title(feature)
####################################################
# TODO: Import 'r2_score'
from sklearn.metrics import r2_score


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)

    # Return the score
    return score
#####################################################