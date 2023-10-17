from transformers import AutoTokenizer, MobileBertForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
model = MobileBertForMaskedLM.from_pretrained("google/mobilebert-uncased")

inputs = tokenizer("The city of [MASK] is the capital of France.", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
print(tokenizer.decode(predicted_token_id))
