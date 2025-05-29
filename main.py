import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using device: {device}")

data_df = pd.read_csv("C:/Users/vinic/OneDrive/Área de Trabalho/py_training/classData/archive/riceClassification.csv")

original_df = data_df.copy()

data_df.dropna(inplace=True)
data_df.drop(['id'], axis=1, inplace=True)

for column in data_df.columns:
    data_df[column] = data_df[column]/data_df[column].abs().max()

x = np.array(data_df.iloc[:, :-1])

y = np.array(data_df.iloc[:, -1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)


#print(f" Train shape: {x_train.shape}\n Test shape: {x_test.shape}\n Validation shape: {x_val.shape}")

class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
training_dataset = dataset(x_train, y_train)
validation_dataset = dataset(x_val, y_val)
testing_dataset = dataset(x_test, y_test)

print(f"Training dataset length: {len(training_dataset)}")
print(f"Validation dataset length: {len(validation_dataset)}")
print(f"Testing dataset length: {len(testing_dataset)}")