import tensorflow as tf
import numpy as np

class PerceptronMulticapaT:
    def __init__(self):
        self.W =[tf.Variable(tf.random.normal([3,32], stddev=0.1,dtype=tf.double),name='W0'),
        tf.Variable(tf.random.normal([32, 2], stddev=0.1,dtype=tf.double),name='W1')]
        self.B =[tf.Variable(tf.zeros([1,32],dtype=tf.double),name="B0"),
        tf.Variable(tf.zeros([1, 2],dtype=tf.double),name="B1")]
        self.Z =[0,0]
        self.A =[0,0]

    def forward(self,x):
        self.Z[0] = tf.matmul(x, self.W[0]) + self.B[0]
        self.A[0] = tf.nn.sigmoid(self.Z[0])
        self.Z[1] = tf.matmul(self.A[0], self.W[1]) + self.B[1]
        self.A[1] = tf.nn.sigmoid(self.Z[1])

    def loss(self,Y):
        return 0.5*np.sum(Y-self.A[1])**2
    
    def predict(self,X,Y):
        self.forward(X)
        real=np.argmax(Y,axis=1)
        output=np.argmax(self.A[1],axis=1)
        accuracy=np.sum(real == output)/Y.shape[0]
        return output,self.A[1],accuracy,self.loss(Y)

    def backward(self,x,y):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
        params=[self.W[0],self.B[0],self.W[1],self.B[1]]
        with tf.GradientTape() as tape:
            self.forward(x)
            loss = tf.compat.v1.losses.mean_squared_error(labels=y,predictions=self.A[1])
        gradients = tape.gradient(loss, params)
        optimizer.apply_gradients(zip(gradients, params))    
    
    def train(self,X,Y,XV=None,YV=None,epochs=10):
        log=open("log.csv","w")
        log.write("epoch;loss;accuracy\n" if XV is None else "epoch;loss;accuracy;accuracy_val\n")
        for i in range(epochs):
            for j in range(X.shape[0]):
                x=tf.convert_to_tensor(X[[j]],dtype=tf.double)
                y=tf.convert_to_tensor(Y[[j]],dtype=tf.double)
                self.forward(x)
                self.backward(x,y)
            output,probs,accuracy,L=self.predict(X,Y)
            status=[str(i),str(L),str(accuracy)]
            if XV is not None:
                output_v,probs_v,accuracy_v,L_v=self.predict(XV,YV)
                status.append(str(accuracy_v))
            print("Iteracion ",i," Loss ",L," Accuracy ",accuracy)
            log.write(";".join(status)+"\n")
        log.close()
    
X=[]
Y=[]
archivo=open("dataset_16_3.csv")
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
mlp=PerceptronMulticapaT()
mlp.train(X,Y,epochs=128)