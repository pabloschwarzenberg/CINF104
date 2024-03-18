import torch
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
import random as rd
from ds import CourseDataset

class PerceptronMulticapaPT:
    def __init__(self):
        self.model=torch.nn.Sequential()
        self.model.append(torch.nn.Linear(3,32))
        self.model.append(torch.nn.Sigmoid())
        self.model.append(torch.nn.Linear(32,1))
        self.model.append(torch.nn.Sigmoid())
        self.criterio=torch.nn.BCELoss()
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=0.01)

    def train(self,x,y,xv,yv):
        self.model.train()
        epochs=128
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred=self.model(x)
            loss=self.criterio(y_pred.squeeze(),y)
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            loss.backward()
            self.optimizer.step()

X=[]
Y=[]
archivo=open("dataset_16_3.csv")
archivo.readline()
for linea in archivo:
    linea=linea.strip().split(";")
    x=list(map(float,[linea[1],linea[2],linea[3]]))
    y=1 if linea[0]=="R" else 0
    X.append(x)
    Y.append(y)
archivo.close()
X=torch.FloatTensor(X)
Y=torch.FloatTensor(Y)
seed=121208
rd.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
mlp=PerceptronMulticapaPT()
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=1)
mlp.train(x_train,y_train,x_val,y_val)