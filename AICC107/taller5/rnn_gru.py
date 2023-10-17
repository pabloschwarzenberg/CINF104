from keras.layers import GRU, Input, Dense
from keras.models import Sequential
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import random as rd

seed=15021949
rd.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

x_train=np.loadtxt("x_train.csv",delimiter="\t",skiprows=0,dtype="float",encoding="utf-8")
y_train=np.loadtxt("y_train.csv",delimiter="\t",skiprows=0,encoding="utf-8")
x_val=np.loadtxt("x_val.csv",delimiter="\t",skiprows=0,dtype="float",encoding="utf-8")
y_val=np.loadtxt("y_val.csv",delimiter="\t",skiprows=0,encoding="utf-8")

model = Sequential()
model.add(Input(name="activity",shape=(60,1)))
model.add(GRU(64,return_sequences=True))
model.add(GRU(64))
model.add(Dense(2, activation='softmax'))
model.summary()

tag="gru_se"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + tag
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['AUC'])
spam_rnn = model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=1, epochs=3,verbose=1,callbacks=[tensorboard_callback])
model.save('models/model_{0}'.format(tag))

yv_pred=model.predict(x_val)
pd.DataFrame(x_val).to_csv(log_dir+"/xv.csv",index=False,sep=";")
pd.DataFrame(yv_pred).to_csv(log_dir+"/yv_pred.csv",index=False,sep=";")
pd.DataFrame(y_val).to_csv(log_dir+"/yv.csv",index=False,sep=";")