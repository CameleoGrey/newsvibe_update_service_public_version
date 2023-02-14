

from pprint import pprint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from src.classes.utils import *
from src.classes.profittm.TreeProfitTM import TreeProfitTM
from src.classes.paths_config import *
from src.classes.profittm.TfidfW2vVectorizer import TfidfW2vVectorizer
from src.classes.profittm.MiniLMVectorizer import MiniLMVectorizer
from src.classes.profittm.DataBaseManager import DataBaseManager
#from networkx.algorithms.centrality import load
print("Start time: {}".format(datetime.now()))


database_manager = DataBaseManager( database_path )
parsed_data = database_manager.get_parsed_data()
parsed_data["parse_datetime"] = pd.to_datetime( parsed_data["parse_datetime"] )
parsed_data = parsed_data[ parsed_data["parse_datetime"] >= datetime(day=7, month=12, year=2022) ]
train_texts = np.unique( parsed_data["content"].to_numpy() )
np.random.seed(45)
np.random.shuffle( train_texts )

######################################
# TF-IDF W2V vectorizer
vectorizer = TfidfW2vVectorizer()
vectorizer.fit(train_texts, vector_size=384, window=5,
            n_jobs=10, min_count=2, sample=1e-5, epochs=100, sg=0, seed=45)
save(vectorizer, os.path.join( interim_dir, "prod_vectorizer.pkl"))

vectorizer = load(os.path.join( interim_dir, "prod_vectorizer.pkl"))
vectorized_texts = vectorizer.vectorize_docs(train_texts, use_tfidf=True, n_jobs=8)
save(vectorized_texts, os.path.join( interim_dir, "vectorized_train_texts.pkl"))
########################################

######################################
# MiniLM vectorizer
"""vectorizer = MiniLMVectorizer()
save(vectorizer, os.path.join( interim_dir, "prod_vectorizer.pkl"))
vectorizer = load(os.path.join( interim_dir, "prod_vectorizer.pkl"))
vectorized_texts = vectorizer.vectorize_docs(train_texts)
save(vectorized_texts, os.path.join( interim_dir, "vectorized_train_texts.pkl"))"""
########################################


vectorized_texts = load(os.path.join( interim_dir, "vectorized_train_texts.pkl"))

##################
# visualize embeddings
#embeddings_subsample = vectorized_texts.copy()
#compressed_x = TSNE(n_jobs=10, verbose=1).fit_transform(embeddings_subsample)
#plt.scatter(compressed_x[:, 0], compressed_x[:, 1], s=1)
#plt.show()
##################

topic_model = TreeProfitTM( max_depth=2 )
topic_model.fit(vectorized_texts)
save( topic_model , os.path.join( interim_dir, "prod_tree_profittm.pkl"), verbose=True)

print("done")
