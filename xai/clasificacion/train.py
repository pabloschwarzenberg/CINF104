import tensorflow as tf
import numpy as np
import random as rd
from ds import CourseDataset
from mlp import PerceptronMulticapaK
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler

seed=121208
rd.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

dataset=CourseDataset("dataset.csv")
X,Y=dataset.getData()

us = RandomOverSampler(random_state=seed)
X,Y=us.fit_resample(X,Y)
Y=Y.reshape(-1,1)

encoder=OneHotEncoder()
encoder.fit(Y)
YT=encoder.transform(Y).todense()

x_train, x_test, y_train, y_test = train_test_split(X, YT, test_size=0.1, stratify=YT, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1, stratify=y_train, random_state=seed)

mlp=PerceptronMulticapaK("oversampling")
mlp.train(x_train,y_train,x_val,y_val)