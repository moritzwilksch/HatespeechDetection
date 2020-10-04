#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
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
from prep_helpers import train_test_val_split
xtrain, xval, xtest, ytrain, yval, ytest = train_test_val_split(clean)

#%%
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
xtrain_cv = cv.fit_transform(xtrain)
xval_cv = cv.transform(xval)
xtest_cv = cv.transform(xtest)

ytrain = np.eye(3)[ytrain]
yval = np.eye(3)[yval]
ytest= np.eye(3)[ytest]

#%%
import tensorflow as tf
from tensorflow import keras

nn = keras.Sequential([
    keras.layers.Dense(input_shape=(len(cv.vocabulary_), ), units=5, activation='tanh'),
    keras.layers.Dense(units=3, activation='softmax')
])

nn.compile('adam', 'CategoricalCrossentropy', metrics=['accuracy'])

#%%
import time
tic = time.time()
history = nn.fit(xtrain_cv.toarray(), ytrain, validation_data=(xval_cv.toarray(), yval), epochs=10, batch_size=32)
tac = time.time()
print(f"Training took {tac - tic:.1f} seconds")
#%%
pd.DataFrame({'train': history.history['loss'], 'validation': history.history['val_loss']}).plot()