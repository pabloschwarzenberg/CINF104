from transformers import TrainingArguments
from transformers import Trainer
from transformers import MobileBertTokenizer
from transformers import MobileBertForSequenceClassification
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import os

os.environ["WANDB_DISABLED"] = "true"

MODEL_MARKER = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_PATH = "./models/models-{0}-".format(MODEL_MARKER)+ "-classifier" 

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

model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)

print(datetime.now()," Loading datasets")

x_train=np.loadtxt("x_train.csv",delimiter="\t",skiprows=0,dtype="str",encoding="utf-8")
y_train=np.loadtxt("y_train.csv",delimiter="\t",skiprows=0,encoding="utf-8")
x_test=np.loadtxt("x_test.csv",delimiter="\t",skiprows=0,dtype="str",encoding="utf-8")
y_test=np.loadtxt("y_test.csv",delimiter="\t",skiprows=0,encoding="utf-8")
x_val=np.loadtxt("x_val.csv",delimiter="\t",skiprows=0,dtype="str",encoding="utf-8")
y_val=np.loadtxt("y_val.csv",delimiter="\t",skiprows=0,encoding="utf-8")

print(datetime.now()," Datasets loaded")

train = DataLoader(x_train.tolist(), y_train.tolist())
test = DataLoader(x_test.tolist(), y_test.tolist())
eval = DataLoader(x_val.tolist(), y_val.tolist())

print(datetime.now()," Datasets tokenized")

print("Start ",datetime.now())

training_args = TrainingArguments(
    evaluation_strategy="epoch",
    save_steps = 100000,
    overwrite_output_dir = True,
    output_dir = OUTPUT_PATH,
#    fp16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
#    gradient_checkpointing=True,
    num_train_epochs= 2.0)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train, eval_dataset=eval
)

trainer.train()
trainer.save_model()

print("End ",datetime.now())