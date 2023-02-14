
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from pprint import pprint
from datetime import datetime

import numpy as np
import os

import matplotlib.pyplot as plt
from src.classes.profittm.TopicInterpreter import TopicInterpreter
from src.classes.profittm.DataBaseManager import DataBaseManager
from src.classes.utils import load, save
from src.classes.paths_config import *
np.random.seed(45)

#########################################################
# fit interpreter on full texts
"""train_texts = load(os.path.join( interim_dir, "prod_train_texts.pkl"))
topic_interpreter = TopicInterpreter()
topic_interpreter.fit(train_texts, vector_size=384, window=5, n_jobs=10, min_count=2, sample=1e-5, epochs=100, sg=0, seed=45)
save( topic_interpreter, os.path.join(interim_dir, "prod_topic_interpreter.pkl") )"""

#########################################################
database_manager = DataBaseManager( database_path )
parsed_data = database_manager.get_parsed_data()
parsed_data["parse_datetime"] = pd.to_datetime( parsed_data["parse_datetime"] )
parsed_data = parsed_data[ parsed_data["parse_datetime"] >= datetime(day=14, month=12, year=2022) ]
test_texts = parsed_data["content"].to_numpy()

#vectorizer = load(os.path.join( interim_dir, "prod_vectorizer.pkl"))
#test_raw_features = vectorizer.vectorize_docs(test_texts, use_tfidf=True, n_jobs=8)
#save( test_raw_features, Path(interim_dir, "test_raw_features.pkl") )
test_raw_features = load( Path(interim_dir, "test_raw_features.pkl") )

test_labels = load( os.path.join(interim_dir, "prod_topic_labels.pkl") )
topic_interpreter = load( os.path.join(interim_dir, "prod_topic_interpreter.pkl") )
topic_model = load( os.path.join( interim_dir, "prod_tree_profittm.pkl") )

#test_topic_features = topic_model.extract_features( test_raw_features)
#print(np.isnan(test_topic_features).sum())

# draw clusters
#topic_interpreter.plot_clusters(test_topic_features, test_labels, level=0, n_jobs=10, plot_path=Path(interim_dir, "clusters_0.jpg"))
#topic_interpreter.plot_clusters(test_topic_features, test_labels, level=1, n_jobs=10, plot_path=Path(interim_dir, "clusters_1.jpg"))
#topic_interpreter.plot_clusters(test_topic_features, test_labels, level=None, n_jobs=10, plot_path=Path(interim_dir, "clusters_2.jpg"))

# draw distances
#topic_interpreter.draw_distances(test_topic_features, test_labels, level=0, plot_path=Path(interim_dir, "distances_0.jpg"))

# extract topics
#topic_names_0 = topic_interpreter.get_topic_names(test_texts, test_labels, level=0, n_jobs=8)
#pprint( topic_names_0 )
#topic_names_1 = topic_interpreter.get_topic_names(test_texts, test_labels, level=1, n_jobs=8)
#pprint( topic_names_1 )

# draw topic tree
group_summaries = load( Path( interim_dir, "prod_group_summaries.pkl" ) )
topic_interpreter.plot_topic_graph(test_texts, test_labels, 
                                   path_to_save = os.path.join(interim_dir, "topic_graph.gv"), 
                                   n_jobs=8, group_summaries=group_summaries)

print("done")
