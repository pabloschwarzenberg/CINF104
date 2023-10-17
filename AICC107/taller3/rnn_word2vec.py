from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from keras.models import Sequential
import numpy as np
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import SimpleRNN, Dense, Input
import pandas as pd
import string
import datetime
import tensorflow as tf

seed=12122008
dataset = open("sms.txt")
X=[]
Y=[]
translator = str.maketrans(dict.fromkeys(string.punctuation))
for linea in dataset:
    label,text=linea.strip().split("\t")
    text=text.translate(translator)
    message=[]
    for i in sent_tokenize(text):
        for j in word_tokenize(i):
            message.append(j.lower())
    X.append(message)
    y=[0,1] if label=="spam" else [1,0]
    Y.append(y)

model = Word2Vec(X, min_count = 1, vector_size = 64, window = 4)
X_v=[]
for message in X:
    message_v=[]
    for word in message:
        message_v.append(model.wv[word])
    X_v.append(message_v)
X_v=pad_sequences(X_v)
X_v=np.asarray(X_v)
Y=np.asarray(Y)
print(X_v.shape)

x_train, x_val, y_train, y_val = train_test_split(X_v, Y, test_size=0.2, stratify=Y, random_state=seed)

print(x_train.shape,y_train.shape)
print(x_val.shape,y_val.shape)

model = Sequential()
model.add(Input(name="message",shape=(x_train.shape[1],x_train.shape[2])))
model.add(SimpleRNN(16))
model.add(Dense(2, activation='softmax'))
model.summary()

tag="rnnw2v"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + tag
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['AUC'])
spam_rnn = model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=1, epochs=2,verbose=2,callbacks=[tensorboard_callback])
model.save('models/model_{0}'.format(tag))

yv_pred=model.predict(x_val)
pd.DataFrame(yv_pred).to_csv("rnnw2v_yv_pred.csv",index=False,sep=";")
pd.DataFrame(y_val).to_csv("rnnw2v_yv.csv",index=False,sep=";")