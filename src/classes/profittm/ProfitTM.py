
from src.classes.utils import save, load
import gc

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from src.classes.utils import save, load
from sklearn.metrics import calinski_harabasz_score
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from scipy.spatial.distance import cosine
import numpy as np
from copy import deepcopy
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from src.classes.profittm.CenterLossCompressor import CenterLossCompressor
from src.classes.profittm.StubClassifier import StubClassifier
from hashlib import md5
from lightgbm import LGBMClassifier
from src.classes.grey_clusterizer.GreyClusterizer import GreyClusterizer
from torch.jit import isinstance

class ProfitTM():

    def __init__(self, SVM_C=1.0, n_jobs=10, verbose=1, name=None):

        self.feature_extractor = CenterLossCompressor()
        #self.classifier = SVC(C=SVM_C)  # SVC(C=0.2)
        self.classifier = SGDClassifier(n_jobs=n_jobs)
        #self.classifier = LGBMClassifier(n_jobs=n_jobs)
        self.topic_names = None
        self.topic_count = None
        self.name = name

        pass
    
    def fit(self, x, max_agg_elements=30000, target_n_clusters=20, opt_param_dev=0.0, n_optimal_steps=1,
            batch_size=20, base_epochs=25):
        
        if not isinstance(x, np.ndarray):
            x = np.array( x )
        
        x = x.copy()
        
        if len(x) <= target_n_clusters:
            target_n_clusters = 1
        if len(x) == 1:
            x = np.vstack([x, x])
        
        if len(x) > max_agg_elements:
            np.random.seed(45)
            subsample_ids = np.random.choice( list(range(len(x))), size=max_agg_elements, replace=False )
            if not isinstance(subsample_ids, np.ndarray):
                subsample_ids = np.array( subsample_ids )
            x = x[subsample_ids]
            
        """self.find_optimal_clusters(
            x,
            target_n_clusters,
            opt_param_dev,
            n_optimal_steps)
        clust_train_y = self.clusterizer.labels_"""
        
        self.clusterizer = AgglomerativeClustering(n_clusters=target_n_clusters, linkage='ward')
        self.clusterizer.fit(x)
        clust_train_y = self.clusterizer.labels_
        
        """clusterizer = GreyClusterizer(input_dim=len(x[0]),
                 latent_size=256, cluster_num=target_n_clusters, hidden_layer_dim=512,
                 instance_feature_dim=256, hidden_layers_num=3, dropout_rate=0.05,
                 device="cuda", checkpoint_dir=None, model_name="grey_clusterizer")
        cluster_batch_size = len(x) // 100 + 1
        if cluster_batch_size > 256:
            cluster_batch_size = 256
        clusterizer.fit(x, epochs = 35, learning_rate = 0.001,
            batch_size=cluster_batch_size, backup_freq=999, warm_up_epochs=5, early_stopping_rounds=5,
            augmentation_power=0.2, instance_temperature=0.5, cluster_temperature=1.0, verbose_freq=10, save_best_in_ram=True)
        clust_train_y = clusterizer.predict_batch_cluster_labels(x, batch_size=256, k_n=None, verbose=True, save_gpu_memory=False)"""
        
        optimal_clust_train_y = self.distance_bazed_cluster_merge(x, clust_train_y, metric='cosine', n_quantiles=20)
        """
        if len(np.unique(optimal_clust_train_y)) <= 2:
            #self.classifier = StubClassifier(stub_y=optimal_clust_train_y[0])
            #return self
            pass
        else:
            clust_train_y = optimal_clust_train_y"""
        
        if len(np.unique(optimal_clust_train_y)) == 1:
            self.classifier = StubClassifier(stub_y=optimal_clust_train_y[0])
            print("Optimal clusters by distance merge is 1. Place stub 1 class classifier.")
            return self
            pass
        else:
            clust_train_y = optimal_clust_train_y

        self.feature_extractor.fit(
            x,
            clust_train_y,
            batch_size=batch_size,
            epochs=base_epochs)
        x_compressor_features = self.feature_extractor.predict(x)
        self.classifier.fit(x_compressor_features, clust_train_y)

        clust_train_y = self.classifier.predict(x_compressor_features)
        optimal_clust_train_y = self.size_bazed_cluster_merge(x_compressor_features, clust_train_y, small_cluster_threshold=0.04, n_quantiles=20)
        
        """if len(np.unique(optimal_clust_train_y)) <= 2:
            #self.classifier = StubClassifier(stub_y=optimal_clust_train_y[0])
            #return self
            pass
        else:
            clust_train_y = optimal_clust_train_y"""
            
        if len(np.unique(optimal_clust_train_y)) == 1:
            #self.classifier = StubClassifier(stub_y=optimal_clust_train_y[0])
            print("Skip cluster size merge")
            return self
            #pass
        else:
            clust_train_y = optimal_clust_train_y
        
        
        self.topic_count = len(np.unique(clust_train_y))
        self.feature_extractor.fit(
            x,
            clust_train_y,
            batch_size=batch_size,
            epochs=base_epochs)
        x_compressor_features = self.feature_extractor.predict(x)
        self.classifier.fit(x_compressor_features, clust_train_y)
        
        return self
    
    def get_topic_names(self, x, current_level):

        x = np.array(x)
        topic_dict = {}
        y = self.predict(x)
        uniq_y = np.unique(y)
        for i in range(len(uniq_y)):
            clust_x = x[y == uniq_y[i]]
            
            topic_names = str(current_level) + "_" + str(uniq_y[i])
            
            topic_dict[uniq_y[i]] = topic_names

        return topic_dict
    
    def get_features(self, x):
        
        estimates = self.feature_extractor.predict(x)
        return estimates

    def get_class_estimates(self, x):
        features = self.get_features(x)
        estimates = self.classifier.decision_function(features)
        return estimates

    def predict(self, x):
        
        if isinstance(self.classifier, StubClassifier):
            pred_y = self.classifier.predict(x)
        else:
            estimates = self.get_features(x)
            pred_y = self.classifier.predict(estimates)
            
        return pred_y

    
    def find_optimal_clusters(
            self, x, target_n_clusters, target_n_clustersDev, n_optimal_steps):

        start_n_clusters = int((target_n_clustersDev) * target_n_clusters) + 1
        end_n_clusters = int((1 + (1 - target_n_clustersDev)) * target_n_clusters)
        n_clust = np.linspace(
            start_n_clusters,
            end_n_clusters,
            n_optimal_steps,
            dtype=int)
        best_clusterizer = None
        best_score = -1e20
        best_n_clusters = None
        for i in range(len(n_clust)):
            clusterizer = AgglomerativeClustering(
                n_clusters=n_clust[i], linkage='ward')
            clusterizer.fit(x)
            labels = clusterizer.labels_
            if len(np.unique(labels)) == 1:
                best_clusterizer = deepcopy(clusterizer)
                best_n_clusters = n_clust[i]
                break
            score = calinski_harabasz_score(x, labels=labels)
            print(
                '{} | n_clusters = {} | score = {}'.format(
                    'Stub', n_clust[i], score))
            if score > best_score:
                best_clusterizer = deepcopy(clusterizer)
                best_score = score
                best_n_clusters = n_clust[i]
        print('Best score at {}: {}'.format(best_n_clusters, best_score))

        self.clusterizer = deepcopy(best_clusterizer)
        pass

    def size_bazed_cluster_merge(
            self, x, y, small_cluster_threshold=0.04, n_quantiles=20):
        
        y = y.copy()
        # get size threshold
        x = np.array(x)
        uniq_y = np.unique(y)
        clust_sizes = []
        for i in range(len(uniq_y)):
            clust_x = x[y == uniq_y[i]]
            clust_size = len(clust_x)
            clust_sizes.append(clust_size)
        clust_sizes = np.array(clust_sizes)
        size_quantiles = pd.DataFrame({'clust_size': clust_sizes})
        print('size quantiles')
        pprint(
            pd.qcut(
                size_quantiles['clust_size'],
                n_quantiles,
                duplicates='drop').value_counts().index)
        size_quantiles = list(
            sorted(
                list(
                    pd.qcut(
                        size_quantiles['clust_size'],
                        n_quantiles,
                        duplicates='drop').value_counts().index)))
        #size_quantiles[0].left = abs(size_quantiles[0].left)

        size_threshold = None
        relative_borders = []
        # last max change can be at the end of sorted quantiles
        # define optimal threshold as the max change
        for i in range(len(size_quantiles) - 1):
            relative_border = size_quantiles[i].right / abs(size_quantiles[i].left)
            relative_borders.append(relative_border)
        # if no changes then don't merge
        if len(relative_borders) == 0:
            return y
        max_relative_border_id = np.argmax(relative_borders)
        max_relative_border = relative_borders[max_relative_border_id]
        # if there was no big change between sorted quantile sizes
        # then there are no small trash clusters
        print('Max relative border: {}'.format(max_relative_border))
        size_threshold = size_quantiles[max_relative_border_id].right

        # find small clusters
        merge_dict = {}
        max_cluster_size = max(clust_sizes)
        relative_sizes = []
        for i in range(len(uniq_y)):
            clust_x = x[y == uniq_y[i]]
            clust_size = len(clust_x)
            relative_size = clust_size / max_cluster_size
            relative_sizes.append(relative_size)
            if clust_size <= size_threshold:
                # if relative_size <= small_cluster_threshold:
                merge_dict[uniq_y[i]] = []

        print('relative sizes')
        pprint(relative_sizes)
        print('Max cluster size: {}'.format(max_cluster_size))
        # find nearest big cluster for small cluster
        for small_cluster_id in merge_dict.keys():
            small_cluster = x[y == small_cluster_id]
            small_cluster_center = np.mean(small_cluster, axis=0)
            min_dist = 1e30
            best_big_cluser_id = None
            for big_cluster_id in uniq_y:
                if small_cluster_id == big_cluster_id:
                    continue

                big_cluster = x[y == big_cluster_id]
                big_cluster_size = len(big_cluster)
                if big_cluster_size <= size_threshold:  # don't merge with other small
                    continue

                big_cluster_center = np.mean(big_cluster, axis=0)
                dist = cosine(small_cluster_center, big_cluster_center)
                if dist < min_dist:
                    best_big_cluser_id = big_cluster_id
            merge_dict[small_cluster_id].append(best_big_cluser_id)
        pprint(merge_dict)

        optimal_y = self.merge_clusters(y, merge_dict)
        return optimal_y

    def distance_bazed_cluster_merge(
            self, x, y, metric='cosine', n_quantiles=20):
            
        # get centers of each cluster as mean of top N words closest to center
        y = y.copy()
        x = np.array(x)
        uniq_y = np.unique(y)
        centers = []
        for i in range(len(uniq_y)):
            clust_x = x[y == uniq_y[i]]
            clusters_center = np.mean(clust_x, axis=0)
            centers.append(clusters_center)
        centers = np.array(centers)

        # get distance threshold for merging
        distances = []
        for i in range(len(centers)):
            for j in range(len(centers)):
                if i == j:
                    continue
                if metric == 'cosine':
                    dist = cosine(centers[i], centers[j])
                else:
                    dist = euclidean(centers[i], centers[j])
                distances.append(dist) 
        if len(distances) == 0:
            return y
        distances = pd.DataFrame({'dist': distances})
        
        pprint(pd.qcut(distances['dist'], n_quantiles, duplicates='drop'))
        distance_quantiles = list(
            sorted(
                list(
                    pd.qcut(
                        distances['dist'],
                        n_quantiles,
                        duplicates='drop').value_counts().index)))

        # if no quantiles then don't merge
        if len(distance_quantiles) == 0:
            return y

        #####################################
        distance_threshold = None
        relative_borders = []
        # last max change can be at the end of sorted quantiles
        # define optimal threshold as the max change
        for i in range(len(distance_quantiles) - 1):
            relative_border = distance_quantiles[i].right / abs(distance_quantiles[i].left)
            relative_borders.append(relative_border)
        # if no changes then don't merge
        if len(relative_borders) == 0:
            return y
        max_relative_border_id = np.argmax(relative_borders)
        max_relative_border = relative_borders[max_relative_border_id]
        # if there was no big change between sorted quantile sizes
        # then there are no small trash clusters
        print('Max relative distance border: {}'.format(max_relative_border))
        distance_threshold = distance_quantiles[max_relative_border_id].right
        #####################################

        #####################################
        #distance_threshold = distance_quantiles[0].right
        #####################################

        # get clusters which centers are closer than distance threshold
        merge_dict = {}
        for i in range(len(centers)):
            merge_dict[uniq_y[i]] = []
            for j in range(len(centers)):
                if i == j:
                    continue
                if metric == 'cosine':
                    dist = cosine(centers[i], centers[j])
                else:
                    dist = euclidean(centers[i], centers[j])
                if dist <= distance_threshold:
                    merge_dict[uniq_y[i]].append(uniq_y[j])
            if len(merge_dict[uniq_y[i]]) == 0:
                merge_dict[uniq_y[i]].append(-1)

        optimal_y = self.merge_clusters(y, merge_dict)
        return optimal_y

    def merge_clusters(self, y, merge_dict):
        # get initial merge components list
        merge_list = []
        for key in merge_dict.keys():
            if -1 not in merge_dict[key]:
                merge_component = []
                merge_component.append(key)
                for cluster_to_merge in merge_dict[key]:
                    merge_component.append(cluster_to_merge)
                merge_component = list(sorted(merge_component))
                merge_list.append(merge_component)
        uniq_components = []
        for merge_component in merge_list:
            if merge_component not in uniq_components:
                uniq_components.append(merge_component)
        merge_list = uniq_components
        merge_list = list(sorted(merge_list, key=len, reverse=True))
        print(merge_list)

        # clean merge list
        components_lens = set()
        for merge_component in merge_list:
            components_lens.add(len(merge_component))
        components_lens = list(sorted(list(components_lens), reverse=True))
        for k in range(len(components_lens[:len(components_lens) - 1])):
            target_components = []
            target_idx = []
            sub_components = []
            subset_idx = []
            for i in range(len(merge_list)):
                if len(merge_list[i]) == components_lens[k]:
                    target_components.append(set(merge_list[i]))
                    target_idx.append(i)
                elif len(merge_list[i]) <= 1:
                    pass
                elif len(merge_list[i]) in components_lens[k + 1:]:
                    sub_components.append(set(merge_list[i]))
            for i in range(len(target_components)):
                for j in range(len(sub_components)):
                    intersect = sub_components[j].intersection(target_components[i])
                    if sub_components[j] == intersect:
                        subset_idx.append(j)
            subset_idx = list(set(subset_idx))
            for i in range(len(target_components)):
                for j in range(len(subset_idx)):
                    target_components[i] = target_components[i].difference(
                        sub_components[subset_idx[j]])
            for i in range(len(target_idx)):
                merge_list[target_idx[i]] = target_components[i]

        for i in range(len(merge_list)):
            merge_list[i] = set(merge_list[i])
        tmp = []
        for i in range(len(merge_list)):
            if merge_list[i] not in tmp and len(merge_list[i]) > 1:
                tmp.append(merge_list[i])
        merge_list = tmp
        print(merge_list)

        # get final clean components
        clean_components = []
        for i in range(len(merge_list)):
            current_component = set()
            for j in range(i, len(merge_list)):
                if current_component == set():
                    current_component = current_component.union(merge_list[j])
                else:
                    if current_component.intersection(merge_list[j]) != set():
                        current_component = current_component.union(
                            merge_list[j])
            not_in_flag = True
            for i in range(len(clean_components)):
                if current_component.intersection(
                        clean_components[i]) != set():
                    not_in_flag = False
            if not_in_flag:
                clean_components.append(current_component)
        print(clean_components)

        # set new labels
        for opt_set_y in clean_components:
            min_y = min(opt_set_y)
            for i in range(len(y)):
                if y[i] in opt_set_y:
                    y[i] = min_y
        uniq_y = np.unique(y)
        old_new_dict = {}
        for i in range(len(uniq_y)):
            old_new_dict[uniq_y[i]] = i
        for i in range(len(y)):
            y[i] = old_new_dict[y[i]]
        optimal_y = y

        uniq_y = np.unique(optimal_y)
        print('Optimal topics count = {}'.format(len(uniq_y)))

        return optimal_y
