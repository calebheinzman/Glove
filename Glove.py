import pickle

from glove import Corpus, Glove

lines = []

fileName = 'GloveTrainingDocument.txt'
with open(fileName, encoding="utf8", newline = '') as txtFile:
    for row in txtFile:
        row = row.rstrip()
        row = row.strip('\n')
        row = row.split(' ')
        lines.append(row)


#Creating a corpus object
corpus = Corpus()

#Training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(lines, window=10)

glove = Glove(no_components=25, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove.model')

dictionary = glove.dictionary
vectors = glove.word_vectors
embeddings = {}
for word in dictionary:
    embeddings[word] = vectors[dictionary[word]]

f = open("dictionary.pkl","wb")
pickle.dump(embeddings,f)
f.close()


