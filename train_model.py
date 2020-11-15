from data import IrisDataSet
from sklearn import tree
from model_iris import ClassifierNN
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch
import pdb

mymodel = ClassifierNN(input_features=4, hidden_layer1=25, hidden_layer2=30, output_features=3)

train_ds = IrisDataSet(file_name='iris.data')

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):
    for X_train, y_train in tqdm(train_loader, desc=f"epochs {i}"):
        y_pred = mymodel.forward(X_train)
        y_train = y_train.to(dtype=torch.long)
        loss = criterion(y_pred, y_train)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

pdb.set_trace()