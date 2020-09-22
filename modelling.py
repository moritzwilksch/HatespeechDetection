#%%
import pandas as pd
from scipy.stats.stats import ttest_1samp
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
from sklearn.metrics import classification_report, confusion_matrix
print('\n' + classification_report(yval, nb.predict(xval_cv)))
print(confusion_matrix(yval, nb.predict(xval_cv)))

#%%
############################################################################
############################################################################
# Dense Net on BOW data
xtrain_cv_tensor = torch.Tensor(xtrain_cv.toarray())
xval_cv_tensor = torch.Tensor(xval_cv.toarray())
xtest_cv_tensor = torch.Tensor(xtest_cv.toarray())

"""eye = np.eye(3)
ytrain_tensor = torch.Tensor(eye[ytrain.values])
yval_tensor = torch.Tensor(eye[yval.values])
ytest_tensor = torch.Tensor(eye[ytest.values])"""

ytrain_tensor = torch.Tensor(ytrain.values).long()
yval_tensor = torch.Tensor(yval.values).long()
ytest_tensor = torch.Tensor(ytest.values).long()
#%%
BATCHSIZE = 32
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtrain_cv_tensor, ytrain_tensor), batch_size=BATCHSIZE, shuffle=True)

#%%
net = nn.Sequential(
    nn.Linear(in_features=len(cv.vocabulary_), out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=3),
    nn.Softmax()
)


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.xavier_uniform(m.weight)

net.apply(init_weights)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters())

#%%
from torch_lr_finder import LRFinder
lrf = LRFinder(net, optim, criterion)
lrf.range_test(train_loader, start_lr=0.0001, end_lr=1)
lrf.plot()
lrf.reset()

#%%
# SUPER SLOW...
torch.set_num_threads(2)
scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=10**-2.5, max_lr=10**-1.9, cycle_momentum=False)
N_EPOCHS = 10

history = {'train': [], 'val': []}
for epoch in range(N_EPOCHS):
    for x, y in train_loader:
        yhat = net(x)
        loss = criterion(yhat, y)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
    
    with torch.no_grad():
        train_loss = criterion(net(xtrain_cv_tensor), ytrain_tensor)

        val_loss = criterion(net(xval_cv_tensor), yval_tensor)
        val_acc = (((net(xval_cv_tensor)>0.5) == yval_tensor.view(-1, 1)).sum())/(xval_cv_tensor.size(0)*1.0)

        history['train'].append(train_loss.item())
        history['val'].append(val_loss.item())

        print(f"Epoch #{epoch:3}: trainloss = {train_loss:.4f} & valloss = {val_loss:.4f} & val_acc = {val_acc:.3f}")

pd.DataFrame(history).plot()

#%%
print('\n' + classification_report(yval, torch.argmax(net(xval_cv_tensor), axis=1)))
print(confusion_matrix(yval, torch.argmax(net(xval_cv_tensor), axis=1)))


#%%
############################################################################
############################################################################
# Embedding net on Sequence data
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# fit on train
tok = Tokenizer()
tok.fit_on_texts(xtrain.values)

# transform train
xtrain_seq = tok.texts_to_sequences(xtrain.values)
xtrain_seq = pad_sequences(xtrain_seq, maxlen=30)

# transform test
xval_seq = tok.texts_to_sequences(xval.values)
xval_seq = pad_sequences(xval_seq, maxlen=30)

#%%
BATCHSIZE = 32
xtrain_seq_tensor = torch.Tensor(xtrain_seq).long()
xval_seq_tensor = torch.Tensor(xval_seq).long()

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtrain_seq_tensor, ytrain_tensor), batch_size=BATCHSIZE, shuffle=True)

#%%
emb_dim = 50

class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=max(tok.index_word)+1, embedding_dim=emb_dim)
        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=(2, emb_dim))
        self.maxpool = nn.MaxPool1d(2)
        self.hidden1 = nn.Linear(in_features=928, out_features=256)
        self.hidden2 = nn.Linear(in_features=256, out_features=48)
        self.relu = nn.ReLU()
        self.out = nn.Linear(in_features=48, out_features=3)
        self.softmax = nn.Softmax()

    def forward(self, x):
        embds = self.embedding(x).view((-1, 1, 30, emb_dim))
        conv = self.conv(embds).flatten(1)
        x = self.relu(self.hidden1(conv))
        x = self.relu(self.hidden2(x))
        out = self.softmax(self.out(x))
        return out

embnet = EmbeddingModel()
embnet.apply(init_weights)
embcriterion = nn.CrossEntropyLoss()
emboptim = torch.optim.Adam(embnet.parameters(), lr=0.001)

#%%
lrf = LRFinder(embnet, emboptim, embcriterion)
lrf.range_test(train_loader, start_lr=0.0001, end_lr=1)
lrf.plot()
lrf.reset()

#%%
# TODO: Performance!!
torch.set_num_threads(2)
embscheduler = torch.optim.lr_scheduler.CyclicLR(emboptim, base_lr=10**-4, max_lr=10**-2, mode='exp_range', step_size_up=len(xtrain)//BATCHSIZE*0.5*len(xtrain), cycle_momentum=False)
N_EPOCHS = 10

history = {'train': [], 'val': []}
for epoch in range(N_EPOCHS):
    for x, y in train_loader:
        yhat = embnet(x)
        loss = embcriterion(yhat, y)

        emboptim.zero_grad()
        loss.backward()
        emboptim.step()
        embscheduler.step()
    
    with torch.no_grad():
        train_loss = embcriterion(embnet(xtrain_seq_tensor), ytrain_tensor)

        val_loss = embcriterion(embnet(xval_seq_tensor), yval_tensor)
        val_acc = (((embnet(xval_seq_tensor)>0.5) == yval_tensor.view(-1, 1)).sum())/(xval_seq_tensor.size(0)*1.0)

        history['train'].append(train_loss.item())
        history['val'].append(val_loss.item())

        print(f"Epoch #{epoch:3}: trainloss = {train_loss:.4f} & valloss = {val_loss:.4f} & val_acc = {val_acc:.3f}")

pd.DataFrame(history).plot()

#%%
print('\n' + classification_report(yval, torch.argmax(embnet(xval_seq_tensor), axis=1)))
print(confusion_matrix(yval, torch.argmax(embnet(xval_seq_tensor), axis=1)))
