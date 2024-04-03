from ds import CourseDataset
from sklearn.model_selection import train_test_split
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

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1, stratify=y_train, random_state=seed)

import autokeras as ak
input = ak.Input()
hidden = ak.DenseBlock()(input)
output = ak.ClassificationHead(num_classes=2,metrics=["AUC"])
model=ak.AutoModel(
        max_trials=32,
        inputs=[input],
        outputs=[output]
)
model.fit(x_train, y_train, validation_data=(x_val,y_val),
          epochs=32, batch_size=1,verbose=2)

tf_model = model.export_model()
tf_model.save("autokeras.keras")