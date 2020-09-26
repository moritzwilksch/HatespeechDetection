#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.modules.linear import Linear
import torch
from torch import nn
np.random.seed(1234)

PATH = "."

df = pd.read_csv(PATH+"/data/labeled_data.csv", dtype={'tweet': 'string'}, ).drop('Unnamed: 0', axis=1)
classes = ['hate_speech', 'offensive_language', 'neither']

# 0 = hate speech, 1 = offensive language, 2 = neither
df['label'] = df['class']
df = df.drop('class', axis=1)

#%%
from prep_helpers import start_pipeline, clean_text

clean = (
    df
    .pipe(start_pipeline)
    .pipe(clean_text)
)

#%%
from nltk import FreqDist
fd = FreqDist([row['tweet'] for _, row in clean.iterrows()])
fd.plot(10, cumulative=False)

#%%
from prep_helpers import train_test_val_split
xtrain, xval, xtest, ytrain, yval, ytest = train_test_val_split(clean)

#%%
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
xtrain_cv = cv.fit_transform(xtrain)
xval_cv = cv.transform(xval)
xtest_cv = cv.transform(xtest)

#%%
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(xtrain_cv, ytrain)

#%%
from sklearn.metrics import classification_report, confusion_matrix
print('\n' + classification_report(yval, nb.predict(xval_cv)))
print(confusion_matrix(yval, nb.predict(xval_cv)))

#%%
from sklearn.linear_model import LogisticRegression
print("LOGISTIC REGRESSION")
lr = LogisticRegression()
lr.fit(xtrain_cv, ytrain)
print('\n' + classification_report(yval, lr.predict(xval_cv)))
print(confusion_matrix(yval, lr.predict(xval_cv)))

#%%
from sklearn.svm import LinearSVC
print("LINEAR SVC")
svc = LinearSVC()
svc.fit(xtrain_cv, ytrain)
print('\n' + classification_report(yval, svc.predict(xval_cv)))
print(confusion_matrix(yval, svc.predict(xval_cv)))

#%%
from sklearn.ensemble import RandomForestClassifier
print("RANDOM FOREST")
rf = RandomForestClassifier(n_jobs=-1)
rf.fit(xtrain_cv, ytrain)

print('\n' + classification_report(yval, rf.predict(xval_cv)))
print(confusion_matrix(yval, rf.predict(xval_cv)))

