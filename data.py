import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

class IrisDataSet(Dataset):
    def __init__(self, file_name):
        file_path = os.path.join('data', file_name)
        columns = [
            'sepal_length',
            'sepal_width',
            'petal_length',
            'petal_width',
            'species'
        ]
        self.df = pd.read_csv(file_path, names=columns)
        self.preprocess()
    def preprocess(self):
        self.df['species'] = self.df['species'].astype('category').cat.codes
        categories = [
            'Iris_setosa',
            'Iris_versicolor',
            'Iris_virginica'
            ]
        self.X = self.df.loc[:,self.df.columns != 'species']
        self.y = self.df.species
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=1)
        self.X_train = self.X_train.to_numpy()
        self.X_train = torch.FloatTensor(self.X_train)
        self.y_train = self.y_train.to_numpy()
        self.y_train = torch.FloatTensor(self.y_train)
        self.X_test = self.X_test.to_numpy()
        self.X_test = torch.FloatTensor(self.X_test)
        self.y_test = self.y_test.to_numpy()
        self.y_test = torch.FloatTensor(self.y_test)
        
    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self, i):
        return self.X_train[i], self.y_train[i]

if __name__ == "__train_model__":
    dataset = IrisDataSet(file_name='iris.data')
    print(len(dataset))
    print(dataset[10])


