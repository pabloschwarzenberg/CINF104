from transformers import MobileBertModel
from transformers import MobileBertTokenizer

model = MobileBertModel.from_pretrained("google/mobilebert-uncased")
model.save_pretrained("./mobilebert-uncased/")
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
tokenizer.save_pretrained('./mobilebert-uncased/')