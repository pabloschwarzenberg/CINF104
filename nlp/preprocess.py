from sklearn.model_selection import train_test_split
import pandas as pd
import string

seed=12122008
X=[]
Y=[]
translator = str.maketrans(dict.fromkeys(string.punctuation))
archivo=open("sms.txt")
for linea in archivo:
    linea=linea.strip().split("\t")
    x=linea[1]
    x=x.translate(translator)
    y=[0,1] if linea[0]=="spam" else [1,0]
    X.append(x)
    Y.append(y)
archivo.close()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1, stratify=y_train, random_state=seed)

pd.DataFrame(x_train).to_csv("x_train.csv",index=False,header=False,sep="\t")
pd.DataFrame(y_train).to_csv("y_train.csv",index=False,header=False,sep="\t")
pd.DataFrame(x_test).to_csv("x_test.csv",index=False,header=False,sep="\t")
pd.DataFrame(y_test).to_csv("y_test.csv",index=False,header=False,sep="\t")
pd.DataFrame(x_val).to_csv("x_val.csv",index=False,header=False,sep="\t")
pd.DataFrame(y_val).to_csv("y_val.csv",index=False,header=False,sep="\t")