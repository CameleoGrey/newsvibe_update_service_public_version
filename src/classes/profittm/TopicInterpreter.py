

import numpy as np
from scipy.spatial.distance import cosine

import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

from src.classes.profittm.TfidfW2vVectorizer import TfidfW2vVectorizer

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from src.classes.utils import plot_graph
import networkx as nx


class TopicInterpreter():
    
    def __init__(self, vectorizer=None):
        
        self.vectorizer = vectorizer
        
        pass
    
    def fit(self, corpus, vector_size=100, window=5,
            n_jobs=10, min_count=2, sample=1e-5, epochs=10, sg=0, seed=45):
        
        self.vectorizer = TfidfW2vVectorizer()
        self.vectorizer.fit(corpus, vector_size, window, n_jobs, min_count, sample, epochs, sg, seed)
        
        pass
    
    def make_topic_dict(self, text_vectors, topic_labels, level, n_jobs=8):
        x = np.array(text_vectors)
        topic_dict = {}
        topic_labels = self.encode_topic_labels(topic_labels, level)
        y = topic_labels
        uniq_y = np.unique(y)
        centers = []
        for i in range(len(uniq_y)):
            clust_x = x[y == uniq_y[i]]
            clusters_center = np.mean(clust_x, axis=0)
            centers.append(clusters_center)
            most_similar = self.vectorizer.w2vModel.wv.most_similar(positive=clusters_center, topn=5)
            
            topic_name = []
            for j in range(len(most_similar)):
                topic_name_part = most_similar[j][0]
                topic_name.append( topic_name_part )
            topic_name = " ".join(topic_name)
            
            topic_dict[uniq_y[i]] = topic_name

        return topic_dict
    
    def get_topic_names(self, texts, topic_labels, level, n_jobs=8):
        
        x = self.vectorizer.vectorize_docs(texts, use_tfidf=True, n_jobs=n_jobs)
        topic_dict = self.make_topic_dict(x, topic_labels, level, n_jobs=8)

        return topic_dict
    
    def plot_topic_graph(self, texts, topic_labels, path_to_save, n_jobs=8, group_summaries=None):
        
        # make dicts
        topic_level_dicts = []
        if group_summaries is None:
            text_vectors = self.vectorizer.vectorize_docs(texts, use_tfidf=True, n_jobs=n_jobs)
            for i in range( topic_labels.shape[1] ):
                current_level_dict = self.make_topic_dict(text_vectors, topic_labels, level=i, n_jobs=n_jobs)
                topic_level_dicts.append( current_level_dict )
        else:
            
            for topic in group_summaries.keys():
                topic_summary = group_summaries[topic]
                topic_summary = topic_summary.split(".")
                topic_summary = ".\n".join(topic_summary)
                group_summaries[topic] = topic_summary
            
            for i in range( topic_labels.shape[1] ):
                topic_level_dicts.append( {} )
            for topic_name in group_summaries.keys():
                splitted_topic_name = topic_name.split("_")
                topic_name_level = len(splitted_topic_name) - 1
                topic_level_dicts[topic_name_level][topic_name] = group_summaries[topic_name]
            
        
        encoded_labels = self.encode_topic_labels(topic_labels, level=topic_labels.shape[1]-1)
        uniq_labels = np.unique( encoded_labels )
        topic_graph_edges = []
        for ul in uniq_labels:
            
            splitted_label = ul.split("_")
            
            if len(splitted_label) == 1:
                v = topic_level_dicts[0][splitted_label[0]]
                edge = (v, v)
                topic_graph_edges.append(edge)
                continue
            
            left_part = splitted_label[0]
            for i in range(len(splitted_label)-1):
                right_part = left_part + "_" + splitted_label[i+1]
                mapped_left_part = topic_level_dicts[i][left_part]
                mapped_right_part = topic_level_dicts[i+1][right_part]
                edge = ( mapped_left_part, mapped_right_part )
                topic_graph_edges.append( edge )
                left_part = right_part
        
        topic_graph = nx.OrderedGraph()
        topic_graph.add_edges_from(topic_graph_edges)
        
        plot_graph(topic_graph, path_to_save)
        
        pass
    
    def plot_clusters(self, text_vectors, topic_labels, level=None, n_jobs=8, plot_path="./tmp.jpg"):
        
        merged_labels = self.encode_topic_labels(topic_labels, level)
        encoded_labels = LabelEncoder().fit_transform(merged_labels)

        compressed_x = TSNE(n_jobs=n_jobs, verbose=1).fit_transform(text_vectors)
        uniq_labels = np.unique(encoded_labels)
        for i in uniq_labels:
            plt.scatter(compressed_x[encoded_labels == i, 0],
                        compressed_x[encoded_labels == i, 1], s=1)
            plt.scatter(np.mean(compressed_x[encoded_labels == i, 0]), np.mean(
                compressed_x[encoded_labels == i, 1]), s=50, c='r')
            plt.text(np.mean(compressed_x[encoded_labels == i, 0]), np.mean(
                compressed_x[encoded_labels == i, 1]), str(i), fontsize=15, c='r')

        plt.savefig(plot_path, dpi=500)

        pass
    
    def encode_topic_labels(self, topic_labels, level=None):
        
        topic_labels = topic_labels.copy()
        
        if level is None:
            level = topic_labels.shape[1] - 1
        
        merged_labels = []
        for i in range(len(topic_labels)):
            merged_label = []
            for j in range(level+1):
                merged_label.append(str(topic_labels[i][j])) 
            merged_label = "_".join(merged_label)
            
            merged_labels.append(merged_label)
        
        merged_labels = np.array( merged_labels )
        
        return merged_labels
        
    
    def draw_distances(self, text_vectors, topic_labels, level, plot_path="./tmp.jpg"):
        
        topic_labels = self.encode_topic_labels(topic_labels, level)
        
        clust_dict = {}
        x = text_vectors
        uniq_y = np.unique(topic_labels)
        centers = []
        for i in range(len(uniq_y)):
            clust_x = x[topic_labels == uniq_y[i]]
            clusters_center = np.mean(clust_x, axis=0)
            centers.append(clusters_center)
        centers = np.array(centers)

        distances = []
        for i in range(len(centers)):
            for j in range(len(centers)):
                if i == j:
                    continue
                dist = cosine(centers[i], centers[j])
                distances.append(dist)
        print(len(distances))
        distances = pd.DataFrame({'dist': distances})
        pprint(pd.qcut(distances['dist'], 20))

        compressed_x = []
        min_dist_labels = []
        for i in range(len(centers)):
            min_dist = 1e30
            min_j = -1
            for j in range(len(centers)):
                if i == j:
                    continue
                dist = cosine(centers[i], centers[j])
                if dist < min_dist:
                    # + 0.1 * np.random.standard_normal(1)[0] * dist #noise is only for visualization
                    min_dist = dist
                    min_j = j
            compressed_x.append([min_dist, min_dist])
            min_dist_labels.append(str(uniq_y[i]) + ' | ' + str(uniq_y[min_j]))
        compressed_x = np.array(compressed_x)

        for i in range(len(min_dist_labels)):
            plt.text(compressed_x[i, 0], compressed_x[i, 1],
                     min_dist_labels[i], fontsize=12)
            plt.scatter(compressed_x[i, 0], compressed_x[i, 1], s=1)
        plt.savefig(plot_path, dpi=500)

        pass
    
    
    """def explain_summaries(self):
        #preprocessor = DataPreproc()
        #vectorizer = load("./vectorizer.pkl")
        #summaryMaker = SummaryMaker(preprocessor = preprocessor, vectorizer=vectorizer)
        #train_news = summaryMaker.transform(train_news[:10000])
        
        ###################################
        cluster_train_x = np.array( cluster_train_x )
        pred_y = AgglomerativeClustering(n_clusters=100).fit_predict(cluster_train_x[:10000])
        compressed_x = TSNE(n_jobs=10, verbose=1).fit_transform(cluster_train_x[:10000])
        uniq_labels = np.unique(pred_y)
        for i in range(len(uniq_labels)):
            plt.scatter(compressed_x[pred_y == i, 0], compressed_x[pred_y == i, 1], s=1)
        plt.show()
        ###########################
        
        train_titles_x = train_titles_x[:10000]
        cluster_train_x = TSNE(n_jobs=10, verbose=1).fit_transform(cluster_train_x[:10000])
        trainX = pd.DataFrame( {"news": train_titles_x, "pointx": cluster_train_x[:,0], "pointy": cluster_train_x[:, 1]} )
        print(trainX.head(5))
        fig = px.scatter(trainX, x="pointx", y="pointy", hover_data=["news"])
        fig.show()"""