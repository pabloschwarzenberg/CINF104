import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
import random as rd
from ds import CourseDataset
from mlp import PerceptronMulticapaK
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

seed=121208
rd.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

dataset=CourseDataset("dataset.csv")
X,Y=dataset.getData()

encoder=OneHotEncoder()
encoder.fit(Y)
YT=encoder.transform(Y).todense()

x_train, x_test, y_train, y_test = train_test_split(X, YT, test_size=0.1, stratify=YT, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1, stratify=y_train, random_state=seed)

mlp=PerceptronMulticapaK("stratified")
mlp.train(x_train,y_train,x_val,y_val)