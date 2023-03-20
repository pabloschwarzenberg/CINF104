import numpy as np

class PerceptronMulticapa:
    def __init__(self,seed=0,features=3,hidden=3,output=2):
        self.rg=np.random.RandomState(seed)
        self.A,self.Z,self.D=[0,0],[0,0],[0,0]
        self.B=[np.zeros(shape=(1,hidden)),np.zeros(shape=(1,output))]
        self.W=[self.rg.normal(loc=0.0,scale=0.1,size=(features,hidden)),
            self.rg.normal(loc=0.0,scale=0.1,size=(hidden,output))]
        self.eta=0.01

    # MSE con n==1 porque el batch size es 1 en nuestro ejemplo
    # El gradiente de la función completa tiene la propiedad
    # que puede aproximarse con la suma de los gradientes de las muestras
    # lo que hace el proceso más eficiente
    def loss(self,Y):
        return 0.5*np.sum(Y-self.A[1])**2

    def sigmoide(self,Z):
        return 1/(1+np.exp(-Z))

    def derivada_sigmoide(self,A):
        return A * (1-A)

    def predict(self,X,Y):
        self.forward(X,Y)
        real=np.argmax(Y,axis=1)
        output=np.argmax(self.A[1],axis=1)
        accuracy=np.sum(real == output)/Y.shape[0]
        return output,self.A[1],accuracy,self.loss(Y)

    def forward(self,X,Y):
        self.Z[0]=np.dot(X,self.W[0])+self.B[0]
        self.A[0]=self.sigmoide(self.Z[0])

        self.Z[1]=np.dot(self.A[0],self.W[1])+self.B[1]
        self.A[1]=self.sigmoide(self.Z[1])
    
    def backward(self,X,Y):
        self.D[1]=(self.A[1]-Y)*self.derivada_sigmoide(self.A[1])
        self.W[1]=self.W[1]-self.eta*np.dot(self.A[0].T,self.D[1])
        self.B[1]=self.B[1]-self.eta*self.D[1]

        self.D[0]=np.dot(self.D[1],self.W[1].T)*self.derivada_sigmoide(self.A[0])
        self.W[0]=self.W[0]-self.eta*np.dot(X.T,self.D[0])
        self.B[0]=self.B[0]-self.eta*self.D[0]

    def train(self,X,Y,XV=None,YV=None,epochs=10):
        log=open("log.csv","w")
        log.write("epoch;loss;accuracy\n" if XV is None else "epoch;loss;accuracy;accuracy_val\n")
        for i in range(epochs):
            for j in range(X.shape[0]):
                self.forward(X[[j]],Y[[j]])
                self.backward(X[[j]],Y[[j]])
            output,probs,accuracy,L=self.predict(X,Y)
            status=[str(i),str(L),str(accuracy)]
            if XV is not None:
                output_v,probs_v,accuracy_v,L_v=self.predict(XV,YV)
                status.append(str(accuracy_v))
            print("Iteracion ",i," Loss ",L," Accuracy ",accuracy)
            log.write(";".join(status)+"\n")
        log.close()