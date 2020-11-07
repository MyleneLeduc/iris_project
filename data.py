import pandas as pd
import os

def read_data():
    file_path = os.path.join('data','iris.data')
    columns = [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'species'
    ]
    df = pd.read_csv(file_path, names=columns)
    df['species'] = df['species'].astype('category').cat.codes
    return df


