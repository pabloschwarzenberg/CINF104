from transformers import Trainer
from transformers import MobileBertTokenizer
from transformers import MobileBertForSequenceClassification
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import os
from torchmetrics import AUROC

os.environ["WANDB_DISABLED"] = "true"

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, sentences=None, labels=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
        self.encodings = self.tokenizer(self.sentences,
                                        max_length=512,
                                        truncation = True,
                                        padding = 'max_length')
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.sentences)
    
    def encode(self, x):
        return self.tokenizer(x, return_tensors = 'pt').to(DEVICE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #cuda x gpu en el 1ero
print (f'Device Availble: {DEVICE}')

print(datetime.now()," Loading datasets")

x_train=np.loadtxt("base_x_train.csv",delimiter="\t",skiprows=0,dtype="str",encoding="utf-8")
x_train=x_train[:,1]
y_train=np.loadtxt("y_train.csv",delimiter="\t",skiprows=0,encoding="utf-8")
x_test=np.loadtxt("base_x_test.csv",delimiter="\t",skiprows=0,dtype="str",encoding="utf-8")
x_test=x_test[:,1]
y_test=np.loadtxt("y_test.csv",delimiter="\t",skiprows=0,encoding="utf-8")
x_val=np.loadtxt("base_x_val.csv",delimiter="\t",skiprows=0,dtype="str",encoding="utf-8")
x_val=x_val[:,1]
y_val=np.loadtxt("y_val.csv",delimiter="\t",skiprows=0,encoding="utf-8")

print(datetime.now()," Datasets loaded")

train = DataLoader(x_train.tolist(), y_train.tolist())
test = DataLoader(x_test.tolist(), y_test.tolist())
eval = DataLoader(x_val.tolist(), y_val.tolist())

print(datetime.now()," Datasets tokenized")

print("Start ",datetime.now())

model = MobileBertForSequenceClassification.from_pretrained("./models/mbert-20230625-010030--classifier", num_labels=2)
trainer = Trainer(
    model=model, train_dataset=train, eval_dataset=eval
)
MODEL_MARKER="20230625-010030"
LOG_DIR = "./logs/fit/mbert-{0}".format(MODEL_MARKER)
auroc=AUROC(task="binary")

predictions = trainer.predict(eval).predictions
y_pred=torch.tensor(np.argmax(predictions,axis=1))
y=torch.tensor(np.argmax(y_val,axis=1))
print("AUC eval=",auroc(y_pred,y).numpy())
pd.DataFrame(predictions).to_csv(LOG_DIR+"/yv_pred.csv",index=False,sep=";")
pd.DataFrame(y_val).to_csv(LOG_DIR+"/yv.csv",index=False,sep=";")

predictions = trainer.predict(test).predictions
y_pred=torch.tensor(np.argmax(predictions,axis=1))
y=torch.tensor(np.argmax(y_test,axis=1))
print("AUC test=",auroc(y_pred,y).numpy())
pd.DataFrame(predictions).to_csv(LOG_DIR+"/yt_pred.csv",index=False,sep=";")
pd.DataFrame(y_test).to_csv(LOG_DIR+"/yt.csv",index=False,sep=";")