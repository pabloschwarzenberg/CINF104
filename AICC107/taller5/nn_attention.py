from keras.layers import Attention,Flatten,Input,Dense
from keras.models import Model
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

d=64
input=Input(name="activity",shape=(60,1))
q = Dense(name="wq",units=d)(input)
k = Dense(name="wk",units=d)(input)
v = Dense(name="wv",units=d)(input)
attention = Attention()([q, v, k])
hidden=Dense(d, activation='relu')(attention)
flatten=Flatten()(hidden)
output=Dense(2, activation='softmax')(flatten)
model=Model(inputs=input,outputs=output)
model.summary()

tag="att_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + tag
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['AUC'])
model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=1, epochs=3,verbose=1,callbacks=[tensorboard_callback])
model.save('models/model_{0}'.format(tag))

yv_pred=model.predict(x_val)
pd.DataFrame(x_val).to_csv(log_dir+"/xv.csv",index=False,sep=";")
pd.DataFrame(yv_pred).to_csv(log_dir+"/yv_pred.csv",index=False,sep=";")
pd.DataFrame(y_val).to_csv(log_dir+"/yv.csv",index=False,sep=";")