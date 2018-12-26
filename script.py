import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing, model_selection

df = pd.read_csv('iris.csv')
df = df.fillna(df.mean())

cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

X = np.array(df[cols])
y = np.array(df['Species'])

Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(Xtrain, ytrain)
accuracy = clf.score(Xtest, ytest)