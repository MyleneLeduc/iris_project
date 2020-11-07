# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:12:45 2020

@author: Mylène
"""

from data import read_data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


df = read_data()

df_setosa = df[df.species==0]
df_versicolor = df[df.species==1]
df_virginica = df[df.species==2]


X = df.loc[:,df.columns != 'species']
y = df.species


### Visualisation des données

figure, axes = plt.subplots(2,2,figsize=(15,8))

axes[0][0].scatter(df_setosa['sepal_length'],df_setosa['sepal_width'])
axes[0][0].scatter(df_versicolor['sepal_length'],df_versicolor['sepal_width'])
axes[0][0].scatter(df_virginica['sepal_length'],df_virginica['sepal_width'])
axes[0][0].set_xlabel('sepal_length (cm)')
axes[0][0].set_ylabel('sepal_width (cm)')

p1 = axes[0][1].scatter(df_setosa['petal_length'],df_setosa['petal_width'],label='Iris_setosa')
p2 = axes[0][1].scatter(df_versicolor['petal_length'],df_versicolor['petal_width'],label='Iris_versicolor')
p3 = axes[0][1].scatter(df_virginica['petal_length'],df_virginica['petal_width'],label='Iris_virginica')
axes[0][1].set_xlabel('petal_length (cm)')
axes[0][1].set_ylabel('petal_width (cm)')

axes[1][0].scatter(df_setosa['sepal_length'],df_setosa['petal_length'])
axes[1][0].scatter(df_versicolor['sepal_length'],df_versicolor['petal_length'])
axes[1][0].scatter(df_virginica['sepal_length'],df_virginica['petal_length'])
axes[1][0].set_xlabel('sepal_length (cm)')
axes[1][0].set_ylabel('petal_width (cm)')

axes[1][1].scatter(df_setosa['sepal_width'],df_setosa['petal_width'])
axes[1][1].scatter(df_versicolor['sepal_width'],df_versicolor['petal_width'])
axes[1][1].scatter(df_virginica['sepal_width'],df_virginica['petal_width'])
axes[1][1].set_xlabel('sepal_length (cm)')
axes[1][1].set_ylabel('petal_width (cm)')


plt.gca().legend(handles=[p1,p2,p3], loc='upper right', bbox_to_anchor=(1.35, 2.2))

plt.show()

### Split the data into training set end test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#X_train = torch.FloatTensor(X)
#X_test = torch.FloatTensor(X_test)
#y_train = torch.LongTensor(y)
#y_test = torch.LongTensor(y_test)

X_train = X_train.to_numpy()
X_train = torch.FloatTensor(X_train)
y_train = y_train.to_numpy()
y_train = torch.FloatTensor(y_train)

X_test = X_test.to_numpy()
X_test = torch.FloatTensor(X_test)
y_test = y_test.to_numpy()
y_test = torch.FloatTensor(y_test)

### Create class

class Model(nn.Module):
    def __init__(self, input_features=4, hidden_layer1=25, hidden_layer2=30, output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features,hidden_layer1)                  
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)                  
        self.out = nn.Linear(hidden_layer2, output_features)      
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

### Define model

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    y_train = y_train.to(dtype=torch.long)
    #y_pred = torch.tensor(y_pred, dtype=torch.long)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()

### Validating and testing the model

preds = []
with torch.no_grad():
    for val in X_test:
        y_hat = model.forward(val)
        preds.append(y_hat.argmax().item())

correct = len(y_test)
for i in range(len(y_test)):
    if y_hat[i]==y_test[i]:
        correct[i] = 1
    else:
        correct[i] = 0

            
breakpoint()






