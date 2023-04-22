from transformers import MobileBertModel
from transformers import MobileBertTokenizer
from transformers import AutoTokenizer, AutoModel

model = MobileBertModel.from_pretrained("google/mobilebert-uncased")
model.save_pretrained("./mobilebert-uncased/")
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
tokenizer.save_pretrained('./mobilebert-uncased/')

#model = BertModel.from_pretrained("bert-base-cased")
#model.save_pretrained("./bert-base-cased")
#tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#model = AutoModel.from_pretrained("bert-base-cased")
#tokenizer.save_pretrained('./bert-base-cased/')
#model.save_pretrained('./bert-base-cased/')
