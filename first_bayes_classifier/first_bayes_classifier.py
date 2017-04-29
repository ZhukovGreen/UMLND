import string

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

_DATA_PATH = r'/mnt/83ce4e68-ad37-4350-9ad5-20ecfcd21ea9/clouds/Dropbox/MyEdu/UMLND/first_bayes_classifier/smsspamcollection'
df = pd.read_table(_DATA_PATH,
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])
df.label = df.label.map({'ham': 0, 'spam': 1})

# documents = ['Hello, how are you!',
#              'Win money, win from home.',
#              'Call me now.',
#              'Hello, Call hello you tomorrow?']
# documents = [(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)))(i) for i in documents]
#
# count_vector = CountVectorizer(lowercase=True,
#                                token_pattern='(?u)\\b\\w\\w+\\b',
#                                stop_words=None)
#
# count_vector.fit(documents)
# doc_array = count_vector.transform(documents).toarray()
#
# frequency_matrix = pd.DataFrame(doc_array, columns=count_vector.get_feature_names())
# print(frequency_matrix)

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()
_ = naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)

df.drop()

print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
