from transformers import pipeline
unmasker = pipeline('fill-mask', model='google/mobilebert-uncased')
print(unmasker("The cat is [MASK]."))

print(unmasker("The dog is [MASK]."))