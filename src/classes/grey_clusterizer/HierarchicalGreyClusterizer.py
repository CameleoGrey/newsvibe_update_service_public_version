
import os
from uuid import uuid4

import numpy as np
from src.classes.grey_clusterizer.GreyClusterizer import GreyClusterizer
from sklearn.preprocessing import LabelEncoder

import pickle

class HierarchicalGreyClusterizer():
    def __init__(self, input_dim,
                 latent_size=[256, 256], cluster_num=[10, 2], hidden_layer_dim=[512, 512],
                 instance_feature_dim=[256, 256], hidden_layers_num=[3, 3], dropout_rate=[0.05, 0.05],
                 device="cuda", save_dir=os.path.join("..", "..", "..", "models"), model_name="hierarchical_grey_clusterizer",
                 curLevel=0, parentsName=None):

        self.input_dim = input_dim
        self.latent_size = latent_size
        self.cluster_num = cluster_num
        self.hidden_layer_dim = hidden_layer_dim
        self.instance_feature_dim = instance_feature_dim
        self.hidden_layers_num = hidden_layers_num
        self.dropout_rate = dropout_rate
        self.label_encoder = LabelEncoder()

        self.save_dir = save_dir
        self.model_name = model_name

        self.clusterizer_tree = []
        self.device = device

        self.node = None
        self.childs = {}
        self.topicNames = None
        self.topicCount = None
        self.maxDepth = len(cluster_num)
        self.curLevel = curLevel
        self.treeName = str(uuid4())

        if curLevel == 0 and parentsName is None:
            self.isRoot = True
        else:
            self.isRoot = False

        pass


    def fit(self, x, batch_size=[256, 256], epochs=[200, 200], learning_rate=[0.0001, 0.0001],
            warm_up_epochs=[2, 2], early_stopping_rounds=[10, 10], backup_freq=[100, 100],
            instance_temperature=[0.5, 0.5], cluster_temperature=[1.0, 1.0],
            augmentation_power=[0.2, 0.2], verbose_freq=[10, 10]):

        if self.curLevel == 0:
            self.node = GreyClusterizer(
                    self.input_dim,
                    latent_size=self.latent_size[self.curLevel],
                    cluster_num=self.cluster_num[self.curLevel],
                    hidden_layer_dim=self.hidden_layer_dim[self.curLevel],
                    instance_feature_dim=self.instance_feature_dim[self.curLevel],
                    hidden_layers_num=self.hidden_layers_num[self.curLevel],
                    dropout_rate=self.dropout_rate[self.curLevel],
                    device=self.device,
                    checkpoint_dir=self.save_dir,
                    model_name="grey_clusterizer_{}_{}".format( self.treeName, 0 )
                )

        self.node.fit(x,
                      epochs=epochs[self.curLevel],
                      learning_rate=learning_rate[self.curLevel],
                      batch_size=batch_size[self.curLevel],
                      warm_up_epochs=warm_up_epochs[self.curLevel],
                      early_stopping_rounds=early_stopping_rounds[self.curLevel],
                      backup_freq=backup_freq[self.curLevel],
                      instance_temperature=instance_temperature[self.curLevel],
                      cluster_temperature=cluster_temperature[self.curLevel],
                      augmentation_power=augmentation_power[self.curLevel],
                      verbose_freq=verbose_freq[self.curLevel],
                      save_best_in_ram=True
                )

        if self.curLevel + 1 < self.maxDepth:
            y = self.node.predict_batch_cluster_labels(x, batch_size=batch_size[self.curLevel], )
            uniqY = np.unique(y)
            topicDocs = {}
            for cluster_id in uniqY:
                topicDocs[cluster_id] = x[y == cluster_id]

                self.childs[cluster_id] = HierarchicalGreyClusterizer(
                    self.input_dim,
                    latent_size=self.latent_size,
                    cluster_num=self.cluster_num,
                    hidden_layer_dim=self.hidden_layer_dim,
                    instance_feature_dim=self.instance_feature_dim,
                    hidden_layers_num=self.hidden_layers_num,
                    dropout_rate=self.dropout_rate,
                    device=self.device,
                    save_dir=self.save_dir,
                    model_name="grey_clusterizer",
                    curLevel=self.curLevel + 1,
                    parentsName=self.treeName
                )

                self.childs[cluster_id].node = GreyClusterizer(
                    self.input_dim,
                    latent_size=self.childs[cluster_id].latent_size[self.curLevel + 1],
                    cluster_num=self.childs[cluster_id].cluster_num[self.curLevel + 1],
                    hidden_layer_dim=self.childs[cluster_id].hidden_layer_dim[self.curLevel + 1],
                    instance_feature_dim=self.childs[cluster_id].instance_feature_dim[self.curLevel + 1],
                    hidden_layers_num=self.childs[cluster_id].hidden_layers_num[self.curLevel + 1],
                    dropout_rate=self.childs[cluster_id].dropout_rate[self.curLevel + 1],
                    device=self.device,
                    checkpoint_dir=self.save_dir,
                    model_name="grey_clusterizer_{}_{}_{}".format( self.childs[cluster_id].treeName, self.curLevel + 1, cluster_id )
                )

        if self.curLevel + 1 < self.maxDepth:
            for cluster_id in topicDocs.keys():
                self.childs[cluster_id].fit(
                    topicDocs[cluster_id],
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    warm_up_epochs=warm_up_epochs,
                    early_stopping_rounds=early_stopping_rounds,
                    backup_freq=backup_freq,
                    instance_temperature=instance_temperature,
                    cluster_temperature=cluster_temperature,
                    augmentation_power=augmentation_power,
                    verbose_freq=verbose_freq
                )

        if self.curLevel == 0:
            print("Fitting label encoder")
            self.fit_label_encoder( x, batch_size=batch_size[self.curLevel] )

        pass

    def predict_batch_cluster_labels(self, x, shared_y=None, current_row_ids=None, batch_size=256,
                                     encode_labels=False, verbose=True, save_gpu_memory=False):

        if shared_y is None:
            shared_y = np.zeros( (x.shape[0], self.maxDepth), dtype=np.int ) - 1
            current_row_ids = np.arange( x.shape[0] )

        if self.curLevel < self.maxDepth:
            current_x = x[ current_row_ids ]
            current_y = self.node.predict_batch_cluster_labels(current_x, batch_size=batch_size,
                                                               verbose=verbose, save_gpu_memory=save_gpu_memory)
            shared_y[ current_row_ids, self.curLevel ] = current_y

            if len(self.childs.keys()) != 0:
                uniqY = np.unique(current_y)
                for cluster_id in uniqY:
                    next_row_ids = current_row_ids[current_y == cluster_id]
                    self.childs[cluster_id].predict_batch_cluster_labels(x, shared_y, next_row_ids, batch_size,
                                                                         verbose=verbose, save_gpu_memory=save_gpu_memory)

        if self.curLevel == 0:

            if encode_labels:
                encoded_labels = self.encode_labels( shared_y )
                shared_y = encoded_labels

            return shared_y

    def predict_one_sample_cluster_label(self, x, shared_y=None, current_row_ids=None, batch_size=256, encode_labels=False, save_gpu_memory=False):

        if len( x.shape ) == 1:
            x = x.copy().reshape( (1, -1) )

        if shared_y is None:
            shared_y = np.zeros( (x.shape[0], self.maxDepth), dtype=np.int ) - 1

        if self.curLevel < self.maxDepth:
            current_y = self.node.predict_one_sample_cluster_label(x, save_gpu_memory)
            shared_y[ 0, self.curLevel ] = current_y

            if len(self.childs.keys()) != 0:
                self.childs[current_y].predict_one_sample_cluster_label(x, shared_y, [0], batch_size)

        if self.curLevel == 0:

            if encode_labels:
                encoded_labels = self.encode_labels( shared_y )
                shared_y = encoded_labels[0]

            return shared_y

    def fit_label_encoder(self, x, batch_size=256):

        y_pred = self.predict_batch_cluster_labels(x, batch_size=batch_size, encode_labels=False)

        encoded_labels = []
        for i in range(y_pred.shape[0]):
            str_label = []
            for j in range(y_pred.shape[1]):
                str_label.append(str(y_pred[i][j]))
            str_label = "_".join(str_label)
            encoded_labels.append(str_label)

        encoded_labels = np.array(encoded_labels).reshape((-1, 1))
        self.label_encoder.fit(encoded_labels)
        return self

    def encode_labels(self, y_pred):
        encoded_labels = []
        for i in range(y_pred.shape[0]):
            str_label = []
            for j in range(y_pred.shape[1]):
                str_label.append(str(y_pred[i][j]))
            str_label = "_".join(str_label)
            encoded_labels.append(str_label)
        encoded_labels = np.array(encoded_labels).reshape((-1, 1))
        encoded_labels = self.label_encoder.make_summaries(encoded_labels)
        encoded_labels = encoded_labels.reshape((-1,))
        return encoded_labels

def save(obj, path, verbose=True):
    if verbose:
        print("Saving object to {}".format(path))
    with open(path, "wb") as obj_file:
        pickle.dump( obj, obj_file, protocol=pickle.HIGHEST_PROTOCOL )
    if verbose:
        print("Object saved to {}".format(path))
    pass


def load(path, verbose=True):
    if verbose:
        print("Loading object from {}".format(path))
    with open(path, "rb") as obj_file:
        obj = pickle.load(obj_file)
    if verbose:
        print("Object loaded from {}".format(path))
    return obj