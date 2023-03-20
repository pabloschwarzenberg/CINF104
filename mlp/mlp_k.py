import tensorflow as tf
from tensorflow import keras
from keras.layers import Input,Dense
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import numpy as np

class PerceptronMulticapaK:
    def __init__(self):
        self.input=Input(shape=(None,3))
        self.hidden=Dense(32,activation="sigmoid")
        self.output=Dense(2,activation="sigmoid")
        self.model=Sequential()
        self.model.add(self.input)
        self.model.add(self.hidden)
        self.model.add(self.output)
        self.model.compile(loss='mean_squared_error', optimizer='SGD',metrics = ["accuracy"])

    def train(self,x,y):
        self.model.fit(x, y, epochs=128, batch_size=1,verbose=2)
        
X=[]
Y=[]
archivo=open("dataset_ejemplo_40_3_16.csv")
archivo.readline()
for linea in archivo:
    linea=linea.strip().split(";")
    x=list(map(float,[linea[1],linea[2],linea[3]]))
    y=[1,0] if linea[0]=="R" else [0,1]
    X.append(x)
    Y.append(y)
archivo.close()
X=np.asarray(X)
Y=np.asarray(Y)
mlp=PerceptronMulticapaK()
mlp.train(X,Y)