from nn import PerceptronMulticapa
import numpy as np
from sklearn.model_selection import train_test_split

X=[]
Y=[]
archivo=open("dataset.csv")
archivo.readline()
for linea in archivo:
    linea=linea.strip().split(";")
    x=list(map(float,linea[1:57]))
    y=[1,0] if linea[0]=="R" else [0,1]
    X.append(x)
    Y.append(y)
archivo.close()
X=np.asarray(X)
Y=np.asarray(Y)
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=1)
mlp=PerceptronMulticapa(features=56,hidden=32)
mlp.train(x_train,y_train,XV=x_val,YV=y_val,epochs=128)