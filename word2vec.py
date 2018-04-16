import gensim
import numpy as np
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models.word2vec import Word2Vec
import glob

######
def test_pre_trained_model(path = '~/Downloads/GoogleNews-vectors-negative300.bin'):
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    ball_embeddings = model['ball']
    print("Len: " + str(len(ball_embeddings)))
    print(ball_embeddings)

    ball_embeddings_matrix = np.asmatrix(ball_embeddings)
    output = np.dot(ball_embeddings_matrix.T, ball_embeddings_matrix)
    print(output.shape)
    print(output)


def generate_corpus(file_collection_path='../wiki/output/pages/*.txt'):
    collection = glob.glob(file_collection_path)
    corpus = list()
    for f in collection:
        c = open(f, 'r')
        content = c.readlines()
        for line in content:
            corpus.append(line.split())

    return corpus


def train_corpus(corpus, file_path='w2v_model'):
    phrases = Phrases(sentences=corpus, min_count=25, threshold=50)
    bigram = Phraser(phrases)
    for index, sentence in enumerate(corpus):
        corpus[index] = bigram[sentence]
    size = 300
    window_size = 2  # sentences weren't too long, so
    epochs = 100
    min_count = 2
    workers = 4

    # train word2vec model using gensim
    model = Word2Vec(corpus, sg=1, window=window_size, size=size,
                     min_count=min_count, workers=workers, iter=epochs, sample=0.01, negative=15)
    model.wv.save_word2vec_format(file_path, binary=True)


if __name__ == '__main__':
    corpus = generate_corpus()
    train_corpus(corpus)
    test_pre_trained_model('w2v_model')