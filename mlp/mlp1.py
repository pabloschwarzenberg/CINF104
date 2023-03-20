from nn import PerceptronMulticapa
import numpy as np

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
mlp=PerceptronMulticapa(hidden=32)
mlp.train(X,Y,epochs=128)
