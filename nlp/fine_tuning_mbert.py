from transformers import TrainingArguments
from transformers import Trainer
from transformers import MobileBertTokenizerFast
from transformers import MobileBertForSequenceClassification
import pandas as pd
import torch
import gc
from datetime import datetime

OUTPUT_PATH = data_utils.OUTPUT_PATH + "-classifier-" + str(data_utils.args.ntsamples)
TEST_PREDICTIONS_FILE = OUTPUT_PATH+"/test-predictions-" + data_utils.MODEL_MARKER
EVAL_PREDICTIONS_FILE = OUTPUT_PATH+"/eval-predictions-" + data_utils.MODEL_MARKER
TEST_GRAPH_FILE_PREFIX  = OUTPUT_PATH+"/test-" + data_utils.MODEL_MARKER 
EVAL_GRAPH_FILE_PREFIX  = OUTPUT_PATH+"/eval-" + data_utils.MODEL_MARKER 

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, sentences=None, labels=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = MobileBertTokenizerFast.from_pretrained("google/mobilebert-uncased/")
        
        if bool(sentences):
            self.encodings = self.tokenizer(self.sentences,
                                            max_length=512,
                                            truncation = True,
                                            padding = 'max_length')
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        if self.labels == None:
            item['labels'] = None
        else:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.sentences)
    
    
    def encode(self, x):
        return self.tokenizer(x, return_tensors = 'pt').to(DEVICE)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #cuda x gpu en el 1ero
print (f'Device Availble: {DEVICE}')

model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased/", num_labels=data_utils.NUM_CLASSES)

print(datetime.now()," Loading train")

train_df, DICT_SIZE = data_utils.getDataset("train")
train_texts = train_df['text'].to_list()
train_labels = train_df['labels'].to_list()

train = DataLoader(train_texts, train_labels)
train_df = train_df.iloc[0:0]
print(datetime.now()," starting gc")
gc.collect()

print(datetime.now()," Loading validation")

eval_df, _ = data_utils.getDataset("validation")
eval_texts = eval_df['text'].to_list()
eval_labels = eval_df['labels'].to_list()

eval = DataLoader(eval_texts, eval_labels)
eval_df = eval_df.iloc[0:0]
print(datetime.now()," starting gc")
gc.collect()

print(datetime.now()," Loading test")

test_df, _ = data_utils.getDataset("test")
test_texts = test_df['text'].to_list()
test_labels = test_df['labels'].to_list()

test = DataLoader(test_texts, test_labels)
test_df = test_df.iloc[0:0]
print(datetime.now()," starting gc")
gc.collect()

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

score, fileOutput, allResults = data_utils.compute_metrics(predictions,eval_labels)
data_utils.generate_graphs(allPredictions=predictions, allLabels=eval_labels, rocFilenamePrefix=EVAL_GRAPH_FILE_PREFIX, colorsNames=None, classesNames=None)
f = open(EVAL_PREDICTIONS_FILE,"w+")
f.write(fileOutput)
f.close()
print('Eval ' + str(data_utils.model_name) + ' ' + str(data_utils.args.ntsamples) + ' ' + str(allResults) + '\n')

predictions = trainer.predict(test).predictions

score, fileOutput, allResults = data_utils.compute_metrics(predictions,test_labels)
data_utils.generate_graphs(allPredictions=predictions, allLabels=test_labels, rocFilenamePrefix=TEST_GRAPH_FILE_PREFIX, colorsNames=None, classesNames=None)
f = open(TEST_PREDICTIONS_FILE,"w+")
f.write(fileOutput)
f.close()
print('Test ' + str(data_utils.model_name) + ' ' + str(data_utils.args.ntsamples) + ' ' + str(allResults) + '\n')