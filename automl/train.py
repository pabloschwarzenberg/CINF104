from ds import CourseDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input,Dense
from keras.models import Sequential
import pandas as pd
import datetime
import numpy as np
import random as rd

seed=121208
rd.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

dataset=CourseDataset("dataset.csv")
X,Y=dataset.getData()

us = RandomOverSampler(random_state=seed)
X,Y=us.fit_resample(X,Y)
Y=Y.reshape(-1,1)
Y=np.asarray(Y)

encoder=OneHotEncoder()
encoder.fit(Y)
YT=encoder.transform(Y).todense()
YT=np.asarray(YT)

x_train, x_test, y_train, y_test = train_test_split(X, YT, test_size=0.1, stratify=YT, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1, stratify=y_train, random_state=seed)

model_tag="manualml"
input=Input(shape=(3,))
hidden=Dense(32,activation="relu")
output=Dense(2,activation="softmax")
model=Sequential()
model.add(input)
model.add(hidden)
model.add(output)
optimizer=keras.optimizers.Adam(learning_rate=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics = ["AUC"])

tag=model_tag+"bs1"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + tag
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(x_train, y_train, validation_data=(x_val,y_val),
          epochs=128, batch_size=1,verbose=2,
          callbacks=[tensorboard_callback])
model.save('modelos/model_{0}.keras'.format(tag))
yv_pred=model.predict(x_val)
pd.DataFrame(x_val).to_csv(log_dir+"/xv_bs1.csv",index=False,sep=";")
pd.DataFrame(yv_pred).to_csv(log_dir+"/yv_pred_bs1.csv",index=False,sep=";")
pd.DataFrame(y_val).to_csv(log_dir+"/yv_bs1.csv",index=False,sep=";")