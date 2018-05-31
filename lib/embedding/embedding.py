#!/Users/xingjian.wang/anaconda3/bin/python
'''
This script performs the task of finding a distributed representation
of amino acid using the continuous skip-gram model with 5 sample negative sampling
'''

from gensim.models import Word2Vec
import pandas as pd
import numpy as np
#import pickle
import os
try:
    CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
    CWDIR = os.getcwd()

min_count = 2
dims = [15,] #defines the embedding dimension
windows = [5,] #defines skip-gram window
allWeights = []

for dim in dims:
  for window in windows:
    print('dim: ' + str(dim) + ', window: ' + str(window))
#    df = pd.read_csv("Proteins.txt", delim_whitespace=True, header=0)
    df = pd.read_csv(CWDIR+"/../../data/train/train_dataset.csv")
    df.columns = ['sequence','HLA','target']

    # remove any peptide with  unknown variables
    df = df[df.sequence.str.contains('X') == False]
    df = df[df.sequence.str.contains('B') == False]

    df = df.sample(frac=1)

    text = list(df.sequence)
    sentences = []
    for aa in range(len(text)):
      sentences.append(list(text[aa]))
    print(len(sentences))
    model = None
    model = Word2Vec(sentences, min_count=min_count, size=dim, window=window, sg=1, iter = 10, batch_words=100)

    vocab = list(model.wv.vocab.keys())
    print(vocab)

    print(model.wv.syn0)

    embeddingWeights = np.empty([len(vocab), dim])

    for i in range(len(vocab)):
      embeddingWeights[i,:] = model[vocab[i]]

    allWeights.append(embeddingWeights)

os.system('rm -f '+CWDIR+'/peptideEmbedding.bin',)
#with open(CWDIR+'/peptideEmbedding.pickle', 'wb') as f:
#    pickle.dump(allWeights, f)

model.save(CWDIR+'/peptideEmbedding.bin')
#model = Word2Vec.load(CWDIR+'/peptideEmbedding.bin')
