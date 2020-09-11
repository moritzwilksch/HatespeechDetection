#%%
import pandas as pd
from scipy.stats.stats import ttest_1samp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1234)

PATH = "."

df = pd.read_csv(PATH+"/data/labeled_data.csv", dtype={'tweet': 'string'}, ).drop('Unnamed: 0', axis=1)
classes = ['hate_speech', 'offensive_language', 'neither']

# 0 = hate speech, 1 = offensive language, 2 = neither
df['label'] = df['class']
df = df.drop('class', axis=1)

#%%
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

eng_sw = stopwords.words('english')

def start_pipeline(data):
    return data.copy()

def clean_text(data):
    stmr = PorterStemmer()

    # replace mentions
    data['tweet'] = data['tweet'].str.replace("@\w+", 'XMENTIONX')

    # replace hashtags
    data['tweet'] = data['tweet'].str.replace("#\w+", 'XHASHTAGX')

    # remove punctuation
    data['tweet'] = data['tweet'].str.replace(f"[{string.punctuation}]", "")

    # to lower
    data['tweet'] = data['tweet'].str.lower()

    # remove stopwords
    data['tweet'] = data['tweet'].apply(lambda twt: " ".join([word for word in twt.split() if word not in eng_sw]))

    # stem words
    data['tweet'] = data['tweet'].apply(lambda twt: stmr.stem(twt))


    return data

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
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(clean.tweet, clean.label, random_state=1234, test_size=0.15)
xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, random_state=1234, test_size=0.15)

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
from sklearn.metrics import classification_report
print('\n' + classification_report(yval, nb.predict(xval_cv)))