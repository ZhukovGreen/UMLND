import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

le = LabelEncoder()
ohe = OneHotEncoder()
my_data = pd.DataFrame(np.array([['cat', 'dog', 'duck', 'ara'], [1, 2, 3, 4], [True, False, False, True]]).T,
                       columns=['animal', 'id', 'available'])

print my_data
print pd.get_dummies(my_data)

my_data = pd.DataFrame([le.fit_transform(my_data[feature]) for feature in my_data])

print my_data

print ohe.fit_transform(my_data)
