import tensorflow as tf
from tensorflow import keras
from keras.layers import Input,Dense
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import datetime
import numpy as np
import random as rd

class PerceptronMulticapaK:
    def __init__(self):
        self.input=Input(shape=(None,3))
        self.hidden=Dense(32,activation="sigmoid")
        self.output=Dense(2,activation="sigmoid")
        self.model=Sequential()
        self.model.add(self.input)
        self.model.add(self.hidden)
        self.model.add(self.output)
        self.optimizer=keras.optimizers.SGD(learning_rate=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer,metrics = ["accuracy"])

    def train(self,x,y,xv,yv):
        tag=datetime.datetime.now().strftime("40_3_%Y%m%d-%H%M%S")
        log_dir = "logs/fit/" + tag
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model.fit(x, y, validation_data=(xv,yv),epochs=128, batch_size=1,verbose=2,callbacks=[tensorboard_callback])
        self.model.save('modelos/model_{0}'.format(tag))
        
X=[]
Y=[]
archivo=open("dataset_40_3.csv")
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
seed=121208
rd.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
mlp=PerceptronMulticapaK()
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=1)
mlp.train(x_train,y_train,x_val,y_val)