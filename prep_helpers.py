import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split


eng_sw = stopwords.words('english')


def start_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    return data.copy()


def clean_text(data: pd.DataFrame) -> pd.DataFrame:
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


def train_test_val_split(clean: pd.DataFrame):
    xtrain, xtest, ytrain, ytest = train_test_split(clean.tweet, clean.label, random_state=1234, test_size=0.15)
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, random_state=1234, test_size=0.15)

    return xtrain, xval, xtest, ytrain, yval, ytest