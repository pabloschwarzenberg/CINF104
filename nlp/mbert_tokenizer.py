from transformers import MobileBertTokenizer, MobileBertModel
import torch

tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
model = MobileBertModel.from_pretrained("google/mobilebert-uncased")

sentences=[
    "Ok thats cool Its  just off either raglan rd or edward rd Behind the cricket ground Gimme ring when ur closeby see you tuesday",
    "Just got to"
]

inputs = tokenizer(sentences, max_length=512,truncation = True,padding = 'max_length',return_tensors="pt")
outputs = model(**inputs)
print(outputs)
last_hidden_states = outputs.last_hidden_state
