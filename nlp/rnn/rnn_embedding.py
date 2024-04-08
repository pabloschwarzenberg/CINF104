from keras.layers import SimpleRNN, Embedding, Dense, Input
from keras.models import Sequential
from keras.layers import TextVectorization
from keras.utils import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import string
import datetime
import tensorflow as tf

seed=12122008
archivo=open("sms.txt")
archivo.readline()
y=[]
translator=str.maketrans("","",string.punctuation)
vocabulario={}
mensajes=[]
for linea in archivo:
    linea=linea.strip()
    indice=linea.find("\t")
    clase=linea[0:indice]
    mensaje_raw=linea[indice+1:]
    if clase.lower()=="spam":
        y.append([0,1])
    else:
        y.append([1,0])
    palabras=mensaje_raw.strip().split()
    mensaje=[]
    for palabra in palabras:
        palabra=palabra.lower()
        palabra=palabra.translate(translator)
        if len(palabra)>0 and not(palabra.isdigit()):
            if palabra in vocabulario:
                vocabulario[palabra]+=1
            else:
                vocabulario[palabra]=1
            mensaje.append(palabra)
    mensajes.append(mensaje)
archivo.close()
palabras_diferentes=list(vocabulario.keys())
vocabulary=len(palabras_diferentes)

palabras=[]
for palabra in vocabulario:
    palabras.append([palabra,vocabulario[palabra]])
palabras=sorted(palabras,key=lambda x: x[1],reverse=True)
frecuentes=[]
for palabra in palabras[0:8]:
    frecuentes.append(palabra[0])
print(frecuentes)

sequences=[]
for mensaje in mensajes:
    sequence=[]
    for palabra in mensaje:
        indice=palabras_diferentes.index(palabra)
        sequence.append(indice)
    sequences.append(sequence)

data = pad_sequences(sequences)
input_length=len(data[0])

print(vocabulary,input_length)
X=np.asarray(data)
Y=np.asarray(y)

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, stratify=y, random_state=seed)

print(x_train.shape,y_train.shape)

model = Sequential()
model.add(Input(shape=(input_length,)))
model.add(Embedding(input_dim=vocabulary,output_dim=64))
model.add(SimpleRNN(32))
model.add(Dense(2, activation='softmax'))

model.summary()

tag="rnn"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + tag
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['AUC'])
spam_rnn = model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=1, epochs=2,verbose=2,callbacks=[tensorboard_callback])
model.save('models/model_{0}.keras'.format(tag))

yv_pred=model.predict(x_val)
pd.DataFrame(x_val).to_csv("xv.csv",index=False,sep=";")
pd.DataFrame(yv_pred).to_csv("yv_pred.csv",index=False,sep=";")
pd.DataFrame(y_val).to_csv("yv.csv",index=False,sep=";")
