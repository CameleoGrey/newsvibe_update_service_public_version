
import gc
import os
import multiprocessing as mp
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from joblib import Parallel, delayed
from copy import deepcopy
import gensim.downloader as api
from src.classes.utils import save, load
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

from src.classes.profittm.SimpleDataPreprocessor import SimpleDataPreprocessor

def identity_tokenizer(text):
    return text


class TfidfW2vVectorizer():
    def __init__(self):
        self.w2vModel = None
        self.w2v_dict = None
        self.tfidf_vectorizer = TfidfVectorizer()  # kaggle_all_the_news
        # self.tfidf_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer,
        # lowercase=False, stop_words=None)  # kaggle_survey_2020
        pass
    
    def fit(self, corpus, vector_size=100, window=5,
            n_jobs=10, min_count=2, sample=1e-5, epochs=10, sg=0, seed=45):
        
        corpus = SimpleDataPreprocessor().prerproc_docs(corpus, n_jobs, remove_stub_strings=False)
        
        print('Calculating TF-IDF weights')
        self.tfidf_vectorizer.fit(corpus)
        
        print('Fitting W2V model')
        self.make_w2v_dict(corpus, vector_size=vector_size, window=window, n_jobs=n_jobs,
                                    min_count=min_count, sample=sample, epochs=epochs, sg=sg, seed=seed)
        self.cache_vectors = None
        self.use_cache_vectors = False
        return self

    def make_w2v_dict(self, docs, vector_size=128, window=5, n_jobs=10,
                    min_count=1, sample=0, epochs=100, sg=0, seed=45):

        docs = list(docs)
        for i in range(len(docs)):
            docs[i] = docs[i].split()

        self.w2vModel = Word2Vec(
            docs,
            vector_size=vector_size,
            window=window,
            workers=n_jobs,
            min_count=min_count,
            sample=sample,
            epochs=epochs,
            sg=sg,
            seed=seed)
        self.w2v_dict = dict(
            zip(self.w2vModel.wv.index_to_key, self.w2vModel.wv.vectors))
        docs = None
        gc.collect()
        pass

    
    def vectorize_docs(self, docs, use_tfidf=True, n_jobs=8):
        
        def vectorize_batch(docs, tfidf_vectorizer, w2v_dict, verbose=True, use_tfidf=True):
            docs = list(docs)
            tfidf_feats = tfidf_vectorizer.transform(docs)
            for i in range(len(docs)):
                docs[i] = docs[i].split()
    
            doc_vectors = []
            if verbose:
                proc_range = tqdm(range(len(docs)), desc='Vectorizing docs')
            else:
                proc_range = range(len(docs))
    
            tfidf_vocab = tfidf_vectorizer.vocabulary_
            for i in proc_range:
                tmp_vector = []
                sentence_tfidf = tfidf_feats[i].toarray()
                for j in range(len(docs[i])):
                    if docs[i][j] in w2v_dict:
                        if use_tfidf:
                            if docs[i][j] not in tfidf_vocab:
                                continue
                            tfidfInd = tfidf_vocab[docs[i][j]]
                            tfidf = sentence_tfidf[0][tfidfInd]
                            tmp_vector.append(tfidf * w2v_dict[docs[i][j]])
                        else:
                            tmp_vector.append(w2v_dict[docs[i][j]])
                if len(tmp_vector) != 0:
                    tmp_vector = np.array(tmp_vector)
                    tmp_vector = np.mean(tmp_vector, axis=0)
                else:
                    tmp_vector = np.zeros(list(w2v_dict.values())[0].shape)
                doc_vectors.append(tmp_vector)
            return doc_vectors
        
        
        docs = SimpleDataPreprocessor().prerproc_docs(docs, n_jobs, remove_stub_strings=False)
        
        tfidf_vectorizers = []
        w2v_dicts = []
        for i in range( n_jobs ):
            tfidf_vectorizers.append( deepcopy(self.tfidf_vectorizer) )
            w2v_dicts.append( deepcopy(self.w2v_dict) )
        
        doc_batches = []
        batch_size = len(docs) // n_jobs
        for i in range(n_jobs - 1):
            doc_batches.append( docs[i*batch_size : (i+1)*batch_size] )
        doc_batches.append( docs[(n_jobs-1)*batch_size:] )
        
        vectorized_docs = Parallel(n_jobs, verbose=10)(delayed(vectorize_batch)\
                                                       (doc, tfidf_vectorizer, w2v_dict, verbose=True, use_tfidf=use_tfidf) \
                                                       for doc, tfidf_vectorizer, w2v_dict  \
                                                       in zip(doc_batches, tfidf_vectorizers, w2v_dicts))
        del doc_batches
        gc.collect()
        
        vectorized_docs = np.vstack( vectorized_docs )
        
        return vectorized_docs
        
