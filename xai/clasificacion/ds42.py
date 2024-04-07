from torch.utils.data import Dataset
import numpy as np

class CourseDataset(Dataset):
    def __init__(self,archivo):
        self.archivo=archivo
        self.X=[]
        self.Y=[]
        archivo=open(archivo)
        archivo.readline()
        for linea in archivo:
            linea=linea.strip().split(";")
            x=list(map(float,[linea[1],linea[2],linea[43]]))
            y=1 if linea[0]=="R" else 0
            self.X.append(x)
            self.Y.append(y)
        archivo.close()
        self.X=np.asarray(self.X)
        self.Y=np.asarray(self.Y).reshape(-1,1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index],self.Y[index]
    
    def getData(self):
        return self.X,self.Y