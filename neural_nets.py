#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.modules.linear import Linear
import torch
from torch import nn
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

#%%
############################################################################
############################################################################
# Dense Net on BOW data
xtrain_cv_tensor = torch.Tensor(xtrain_cv.toarray())
xval_cv_tensor = torch.Tensor(xval_cv.toarray())
xtest_cv_tensor = torch.Tensor(xtest_cv.toarray())

ytrain_tensor = torch.Tensor(ytrain.values).long()
yval_tensor = torch.Tensor(yval.values).long()
ytest_tensor = torch.Tensor(ytest.values).long()
#%%
BATCHSIZE = 32
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtrain_cv_tensor, ytrain_tensor), batch_size=BATCHSIZE, shuffle=True)

#%%
net = nn.Sequential(
    nn.Linear(in_features=len(cv.vocabulary_), out_features=5),
    nn.Tanh(),
    nn.Linear(in_features=5, out_features=3),
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
torch.set_num_threads(4)
scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=10**-3, max_lr=10**-1.9, cycle_momentum=False)
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
