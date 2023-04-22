from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

seed=12122008
X=[]
Y=[]
archivo=open("sms.txt")
for linea in archivo:
    linea=linea.strip().split("\t")
    x=linea[1]
    y=1 if linea[0]=="spam" else 0
    X.append(x)
    Y.append(y)
archivo.close()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=seed)
ds_train=[]
for i in range(len(x_train)):
    ds_train.append([y_train[i],x_train[i]])
ds_test=[]
for i in range(len(x_test)):
    ds_test.append([y_test[i],x_test[i]])
pd.DataFrame(ds_test).to_csv("ds_test.csv",index=False,header=False,sep="\t")
pd.DataFrame(ds_train).to_csv("ds_train.csv",index=False,header=False,sep="\t")