import pandas
from sklearn import preprocessing

# creating sample data
sample_data = {'name': ['Ray', 'Adam', 'Jason', 'Varun', 'Xiao'],
               'health': ['fit', 'slim', 'obese', 'fit', 'slim']}
# storing sample data in the form of a dataframe
data = pandas.DataFrame(sample_data, columns=['name', 'health'])

label_encoding = preprocessing.LabelEncoder()

label_encoding.fit(data['health'])

print data

ohe = preprocessing.OneHotEncoder()

label_encoding = label_encoding.fit_transform(data['health'])
print label_encoding
# print label_encoding.reshape(-1, 1)
print ohe.fit_transform(label_encoding.reshape(-1, 1))
