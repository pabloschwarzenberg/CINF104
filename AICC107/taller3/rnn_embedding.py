from keras.layers import SimpleRNN, Embedding, Dense
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import string
import datetime
import tensorflow as tf

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

X=np.asarray(X)
Y=np.asarray(Y)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
words=list(tokenizer.word_index.keys())
print(words[0:10])
vocabulary=len(words)+1

data = pad_sequences(sequences)
input_length=len(data[0])

print(vocabulary,input_length)

x_train, x_val, y_train, y_val = train_test_split(data, Y, test_size=0.2, stratify=Y, random_state=seed)

print(x_train.shape,y_train.shape)

model = Sequential()
model.add(Embedding(input_dim=vocabulary,output_dim=64,input_length=input_length))
model.add(SimpleRNN(16))
model.add(Dense(2, activation='softmax'))

model.summary()

tag="rnn"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + tag
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['AUC'])
spam_rnn = model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=1, epochs=2,verbose=2,callbacks=[tensorboard_callback])
model.save('models/model_{0}'.format(tag))

yv_pred=model.predict(x_val)
pd.DataFrame(x_val).to_csv("xv.csv",index=False,sep=";")
pd.DataFrame(yv_pred).to_csv("yv_pred.csv",index=False,sep=";")
pd.DataFrame(y_val).to_csv("yv.csv",index=False,sep=";")
