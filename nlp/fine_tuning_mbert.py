from transformers import TrainingArguments
from transformers import Trainer
from transformers import MobileBertTokenizer
from transformers import MobileBertForSequenceClassification
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import data_utils
import os

os.environ["WANDB_DISABLED"] = "true"

OUTPUT_PATH = data_utils.OUTPUT_PATH + "-classifier-" + str(data_utils.args.ntsamples)
TEST_PREDICTIONS_FILE = OUTPUT_PATH+"/test-predictions-" + data_utils.MODEL_MARKER
EVAL_PREDICTIONS_FILE = OUTPUT_PATH+"/eval-predictions-" + data_utils.MODEL_MARKER
TEST_GRAPH_FILE_PREFIX  = OUTPUT_PATH+"/test-" + data_utils.MODEL_MARKER 
EVAL_GRAPH_FILE_PREFIX  = OUTPUT_PATH+"/eval-" + data_utils.MODEL_MARKER 

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

model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=data_utils.NUM_CLASSES)

print(datetime.now()," Loading datasets")

x_train=np.loadtxt("x_train.csv",delimiter="\t",skiprows=0,dtype="str")
y_train=np.loadtxt("y_train.csv",delimiter="\t",skiprows=0)
x_test=np.loadtxt("x_test.csv",delimiter="\t",skiprows=0,dtype="str")
y_test=np.loadtxt("y_test.csv",delimiter="\t",skiprows=0)
x_val=np.loadtxt("x_val.csv",delimiter="\t",skiprows=0,dtype="str")
y_val=np.loadtxt("y_val.csv",delimiter="\t",skiprows=0)

print(datetime.now()," Datasets loaded")
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)

train = DataLoader(x_train.tolist(), y_train.tolist())
test = DataLoader(x_test.tolist(), y_test.tolist())
test = DataLoader(x_val.tolist(), y_val.tolist())

print(datetime.now()," Datasets tokenized")

print("Start ",datetime.now())

print (train.__getitem__(0))

training_args = TrainingArguments(
    evaluation_strategy="epoch",
    save_steps = 100000,
    overwrite_output_dir = True,
    output_dir = OUTPUT_PATH,
#    fp16=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
#    gradient_checkpointing=True,
    num_train_epochs= 2.0)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train, eval_dataset=eval
)

trainer.train()
trainer.save_model()

print("End ",datetime.now())

predictions = trainer.predict(eval).predictions

score, fileOutput, allResults = data_utils.compute_metrics(predictions,y_val)
data_utils.generate_graphs(allPredictions=predictions, allLabels=y_val, rocFilenamePrefix=EVAL_GRAPH_FILE_PREFIX, colorsNames=None, classesNames=None)
f = open(EVAL_PREDICTIONS_FILE,"w+")
f.write(fileOutput)
f.close()
print('Eval ' + str(data_utils.model_name) + ' ' + str(data_utils.args.ntsamples) + ' ' + str(allResults) + '\n')

predictions = trainer.predict(test).predictions

score, fileOutput, allResults = data_utils.compute_metrics(predictions,y_test)
data_utils.generate_graphs(allPredictions=predictions, allLabels=y_test, rocFilenamePrefix=TEST_GRAPH_FILE_PREFIX, colorsNames=None, classesNames=None)
f = open(TEST_PREDICTIONS_FILE,"w+")
f.write(fileOutput)
f.close()
print('Test ' + str(data_utils.model_name) + ' ' + str(data_utils.args.ntsamples) + ' ' + str(allResults) + '\n')