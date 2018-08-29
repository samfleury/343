from SRNClassifier import SRNClassifier
import pickle
import gzip
import os
import inspect
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

def shift(xs, n):
    e = np.empty_like(xs)
    if n>= 0:
        e[:n] = np.nan

workingDir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(inspect.currentframe())
    )
)
file = open('serenitynow.txt', 'r')
text = file.read()
file.close()

text = text.split()

le = preprocessing.LabelEncoder()
data = le.fit_transform(text)
print(data[0])

enc = preprocessing.OneHotEncoder()
data = np.expand_dims(data, axis = 1)
enc.fit(data)
data = enc.transform(data).toarray()


x = data
y = np.roll(x, -1 ,axis=0)

clf = SRNClassifier(alpha=1e-2, hidden_layer_size=32, activation='tanh', max_iter=100, verbose= True)
clf.fit(x, y)

yhat = np.empty_like(y)
yhat[0] = x[0]
for n in range(y.shape[0] - 1):
    yhat[n + 1] = clf.predict(yhat[n])

decoded = yhat.dot(enc.active_features_).astype(int)
decoded = le.inverse_transform(decoded)
print(decoded)

file_object = open("outtext.txt", 'w')
for n in range(len(decoded)):
    file_object.write(decoded[n] + ' ')

filename = 'output.txt'

with gzip.open(filename, 'w') as f:
    pickle.dump(clf,f)