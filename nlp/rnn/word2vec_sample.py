from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

word_vectors=KeyedVectors.load("word2vec-google-news-300.model")

#si a king le restamos man y le sumamos woman, que obtenemos?
result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
print(*result[0])

#analogia japan -> japanese, france -> ?
result = word_vectors.most_similar(positive=['france', 'japanese'], negative=['japan'])
print(*result[0])

words=["french","france","italy"]
w0=word_vectors[words[0]]
w1=word_vectors[words[1]]
w2=word_vectors[words[2]]
print("Dimension del vector: ",w0.shape)
print("{0} - {1}".format(words[0],words[1]),cosine_similarity(w0.reshape(1,-1),w1.reshape(1,-1)))
print("{0} - {1}".format(words[0],words[2]),cosine_similarity(w0.reshape(1,-1),w2.reshape(1,-1)))
print("{0} - {1}".format(words[1],words[2]),cosine_similarity(w1.reshape(1,-1),w2.reshape(1,-1)))

words = word_vectors.most_similar(w0-w1+w2)
for word in words[0:3]:
    print(*word)