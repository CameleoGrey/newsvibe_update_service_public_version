
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from gensim.models import Word2Vec
import gensim.downloader as api
from src.classes.utils import save, load
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)


class W2VVectorizer():
    def __init__(self):
        self.w2v_model = None
        self.w2v_dict = None
        pass
    
    def fit(self, corpus, vector_size=100, window=5,
            n_jobs=10, min_count=2, sample=1e-5, epochs=10, sg=0, seed=45):
        self.make_w2v_dict(corpus, vector_size=vector_size, window=window, n_jobs=n_jobs,
                                    min_count=min_count, sample=sample, epochs=epochs, sg=sg, seed=seed)
        return self

    def make_w2v_dict(self, docs, size=128, window=5, n_jobs=10,
                    min_count=1, sample=0, iter=100, sg=0, seed=45):

        docs = deepcopy(docs)
        docs = list(docs)
        for i in range(len(docs)):
            docs[i] = docs[i].split()

        self.w2v_model = Word2Vec(
            docs,
            size=size,
            window=window,
            workers=n_jobs,
            min_count=min_count,
            sample=sample,
            iter=iter,
            sg=sg,
            seed=seed)
        self.w2v_dict = dict(
            zip(self.w2v_model.wv.index2word, self.w2v_model.wv.syn0))
        pass

    def set_model(self, model):
        self.w2v_model = model
        self.w2v_dict = dict(zip(model.index2word, model.vectors))

    def vectorize_docs(self, gramms_docs, n_jobs=1,
                      tfidf_vectorizer=None, verbose=True):

        def process_batch(batch_gramms_docs, tfidf_feats,
                          tfidf_vocab, w2v_dict, verbose=True):
            batchVectors = []

            if verbose:
                proc_range = tqdm(
                    range(
                        len(batch_gramms_docs)),
                    desc='Vectorizing docs')
            else:
                proc_range = range(len(batch_gramms_docs))

            for i in proc_range:
                tmp_vector = []

                sentence_tfidf = tfidf_feats[i].toarray()
                for j in range(len(batch_gramms_docs[i])):
                    if batch_gramms_docs[i][j] in w2v_dict:
                        if tfidf_vectorizer is not None:
                            if batch_gramms_docs[i][j] not in tfidf_vocab:
                                continue
                            tfidf_ids = tfidf_vocab[batch_gramms_docs[i][j]]
                            tfidf = sentence_tfidf[0][tfidf_ids]
                            tmp_vector.append(
                                tfidf * w2v_dict[batch_gramms_docs[i][j]])
                        else:
                            tmp_vector.append(w2v_dict[batch_gramms_docs[i][j]])
                    # else:
                    #    print(batch_gramms_docs[i][j])
                if len(tmp_vector) != 0:
                    tmp_vector = np.array(tmp_vector)
                    tmp_vector = np.mean(tmp_vector, axis=0)
                else:
                    tmp_vector = np.zeros(list(w2v_dict.values())[0].shape)
                batchVectors.append(tmp_vector)
            return batchVectors

        splitted_docs = deepcopy(gramms_docs)
        splitted_docs = list(splitted_docs)

        if n_jobs > 1:
            for i in range(len(splitted_docs)):
                splitted_docs[i] = splitted_docs[i].split()
            splitted_docs = np.array_split(splitted_docs, n_jobs)

            w2v_dicts = []
            for i in range(n_jobs):
                w2v_dicts.append(deepcopy(self.w2v_dict))

            self.w2v_dict = None
            del self.w2v_dict
            self.idf_gramm_dict = None
            del self.idf_gramm_dict

            doc_vectors = Parallel(n_jobs)(delayed(process_batch)(batch_gramms_docs, w2v_dict)
                                          for batch_gramms_docs, w2v_dict in zip(splitted_docs, w2v_dicts))
            self.w2v_dict = deepcopy(w2v_dicts[0])
            doc_vectors = np.vstack(doc_vectors)
        else:
            tfidf_feats = tfidf_vectorizer.make_summaries(splitted_docs)
            #tfidf_feats = tfidf_feats.toarray()
            for i in range(len(splitted_docs)):
                splitted_docs[i] = splitted_docs[i].split()
            doc_vectors = process_batch(
                splitted_docs,
                tfidf_feats,
                tfidf_vectorizer.vocabulary_,
                self.w2v_dict,
                verbose=verbose)

        return doc_vectors

    def set_pretrained_model(self, name):
        info = api.info()  # show info about available models/datasets
        model = api.load('glove-wiki-gigaword-100')
        save('./gwg.pkl', model)
        model = load('./gwg.pkl')
        self.set_model(model)
        pass
    
