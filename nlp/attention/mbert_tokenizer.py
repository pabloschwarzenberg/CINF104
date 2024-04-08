from transformers import MobileBertTokenizer, MobileBertModel

tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
model = MobileBertModel.from_pretrained("google/mobilebert-uncased")

sentences=[
    "Sorry Ill call later in meeting",
    "Just got to"
]

inputs = tokenizer(sentences, max_length=512,truncation = True,padding = 'max_length',return_tensors="pt")
outputs = model(**inputs)
print(outputs)
last_hidden_states = outputs.last_hidden_state
