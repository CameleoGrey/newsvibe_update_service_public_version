from src.classes.profittm.ProfitTM import ProfitTM
import numpy as np
import pandas as pd
from src.classes.utils import save, load
import networkx as nx
import uuid
from src.classes.utils import plot_graph
from sklearn.preprocessing import OneHotEncoder
from numpy import dtype


class TreeProfitTM():

    def __init__(self, max_depth=None, current_level=0, parents_name=None):
        self.node = None
        self.childs = {}
        self.topic_names = None
        self.topic_count = None
        self.max_depth = max_depth
        self.current_level = current_level
        self.tree_name = str(uuid.uuid4())

        if current_level == 0 and parents_name is None:
            self.isRoot = True
        else:
            self.isRoot = False

        pass

    def fit(self, x):

        if self.current_level == 0:
            self.node = ProfitTM()
            self.node.fit(x)
        else:
            self.node.fit(x)
        
        self.topic_names = self.node.get_topic_names(x, current_level=self.current_level)
        self.topic_count = self.node.topic_count

        if self.current_level + 1 < self.max_depth:
            y = self.node.predict(x)
            uniq_y = np.unique(y)
            topic_docs = {}
            for topic in uniq_y:

                topic_docs[topic] = []
                for i in range(len(y)):
                    if y[i] == topic:
                        topic_docs[topic].append(x[i])

                self.childs[topic] = TreeProfitTM(self.max_depth, self.current_level + 1)
                self.childs[topic].node = ProfitTM()

        if self.current_level + 1 < self.max_depth:
            for topic in topic_docs.keys():
                self.childs[topic].fit(topic_docs[topic])
        pass

    def predict(self, x, return_vectors=False):

        shared_predicts = self.prepare_to_predict(x)
        all_row_ids = [i for i in range(len(x))]
        all_row_ids = np.array( all_row_ids )
        shared_predicts = self.hierarchical_predict(x, all_row_ids, shared_predicts, pred_ids=None)

        if return_vectors:
            shared_predicts = self.convert_predicts_to_vectors(shared_predicts)

        return shared_predicts

    def prepare_to_predict(self, x):
        
        shared_predicts = np.zeros( shape=(len(x), self.max_depth), dtype=np.int32 )
        shared_predicts = shared_predicts + np.nan
        
        return shared_predicts

    def hierarchical_predict(self, text_vectors, all_row_ids, shared_predicts, pred_ids=None):

        if self.current_level < self.max_depth:

            if pred_ids is None:
                next_text_vectors_batch = text_vectors
            else:
                next_text_vectors_batch = text_vectors[pred_ids]
            y = self.node.predict(next_text_vectors_batch)
            
            
            if self.current_level == 0:
                shared_predicts[:, 0] = y
            else:
                shared_predicts[pred_ids, self.current_level] = y
                
            uniq_y = np.unique(y)
            for topic in uniq_y:
                
                if pred_ids is None:
                    next_ids = all_row_ids[y == topic]
                else:
                    next_ids = all_row_ids[pred_ids][y == topic]
                
                if len(self.childs.keys()) == 0:
                    #self.leaf_predict(text_vectors, shared_predicts, next_ids)
                    pass
                else:
                    self.childs[topic].hierarchical_predict(text_vectors, all_row_ids, shared_predicts, next_ids)
        
        return shared_predicts
    
    def extract_features(self, x):

        shared_predicts = self.prepare_to_feature_extraction(x)
        all_row_ids = [i for i in range(len(x))]
        all_row_ids = np.array( all_row_ids )
        shared_predicts = self.hierarchical_feature_extraction(x, all_row_ids, shared_predicts, pred_ids=None)

        return shared_predicts

    def prepare_to_feature_extraction(self, x):
        
        shared_predicts = np.zeros( shape=(len(x), self.max_depth * self.node.feature_extractor.latent_dim), dtype=np.float64 )
        shared_predicts = shared_predicts + np.nan
        
        return shared_predicts

    def hierarchical_feature_extraction(self, text_vectors, all_row_ids, shared_predicts, pred_ids=None):

        if self.current_level < self.max_depth:

            if pred_ids is None:
                next_text_vectors_batch = text_vectors
            else:
                next_text_vectors_batch = text_vectors[pred_ids]
                
            y = self.node.predict(next_text_vectors_batch)
            topic_level_features = self.node.get_features(next_text_vectors_batch)
            
            insert_size = self.node.feature_extractor.latent_dim
            if self.current_level == 0:
                shared_predicts[:, : insert_size] = topic_level_features
            else:
                shared_predicts[pred_ids, self.current_level * insert_size : (self.current_level + 1) * insert_size] = topic_level_features
                
            uniq_y = np.unique(y)
            for topic in uniq_y:
                
                if pred_ids is None:
                    next_ids = all_row_ids[y == topic]
                else:
                    next_ids = all_row_ids[pred_ids][y == topic]
                
                if len(self.childs.keys()) == 0:
                    #self.leaf_predict(text_vectors, shared_predicts, next_ids)
                    pass
                else:
                    self.childs[topic].hierarchical_feature_extraction(text_vectors, all_row_ids, shared_predicts, next_ids)
        
        return shared_predicts

    """def leaf_predict(self, text_vectors, shared_predicts, pred_ids):
        docs = text_vectors[pred_ids]
        y = self.node.predict(docs)
        shared_predicts[pred_ids, self.current_level] = y
        pass"""



