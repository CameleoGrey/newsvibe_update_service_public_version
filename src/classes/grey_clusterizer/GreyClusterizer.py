
import os
import gc
import pickle
from datetime import datetime

import numpy as np
from tqdm import tqdm

from src.classes.grey_clusterizer.ClusterNetwork import ClusterNetwork
from src.classes.grey_clusterizer.ClusterLoss import ClusterLoss
from src.classes.grey_clusterizer.InstanceLoss import InstanceLoss

import torch
from torch.utils.data import DataLoader
from copy import deepcopy

from torch.utils.data import Dataset

class GreyClusterizerDataset(Dataset):
    def __init__(self, x_array):
        self.x = np.array(x_array).copy()
        self.x = self.x.astype(np.float32)
        pass

    def __getitem__(self, id):
        x = self.x[id].copy()
        return x

    def __len__(self):
        dataset_len = len(self.x)
        return dataset_len

class GreyClusterizer():
    def __init__(self, input_dim,
                 latent_size=256, cluster_num=10, hidden_layer_dim=512,
                 instance_feature_dim=256, hidden_layers_num=3, dropout_rate=0.05,
                 device="cuda", checkpoint_dir=os.path.join("..", "..", "..", "models"), model_name="grey_clusterizer"):

        self.input_dim = input_dim
        self.instance_feature_dim = instance_feature_dim
        self.device = device
        self.latent_size = latent_size
        self.cluster_num = cluster_num

        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name


        self.encoder = ClusterNetwork(input_dim, latent_size=self.latent_size, hidden_layer_dim=hidden_layer_dim,
                                      hidden_layers_num=hidden_layers_num, dropout_rate=dropout_rate,
                                      instance_feature_dim=self.instance_feature_dim,
                                      cluster_num=self.cluster_num, device=device)

        self.encoder = self.encoder.to(self.device)

        pass

    def augment_batch(self, x, augmentation_power=0.2):

        x_i = x.clone()
        x_j = x.clone()

        eps = np.random.uniform(size=1)
        if eps < augmentation_power:
            replace_ids_count = int(augmentation_power * x_i.shape[1])
            random_column_ids = np.random.randint(low=0, high=x_i.shape[1], size=2 * replace_ids_count)
            ids_to_replace_left = random_column_ids[:replace_ids_count]
            ids_to_replace_right = random_column_ids[replace_ids_count:]
            tmp = x_i.clone()
            x_i[:, ids_to_replace_left] = x_i[:, ids_to_replace_right]
            x_i[:, ids_to_replace_right] = tmp[:, ids_to_replace_left]

            replace_ids_count = int(augmentation_power * x_j.shape[1])
            random_column_ids = np.random.randint(low=0, high=x_j.shape[1], size=2 * replace_ids_count)
            ids_to_replace_left = random_column_ids[:replace_ids_count]
            ids_to_replace_right = random_column_ids[replace_ids_count:]
            tmp = x_j.clone()
            x_j[:, ids_to_replace_left] = x_j[:, ids_to_replace_right]
            x_j[:, ids_to_replace_right] = tmp[:, ids_to_replace_left]

        mean = torch.Tensor([0.0]).to(self.device)
        std = torch.Tensor([1.0]).to(self.device)
        x_i = augmentation_power * torch.normal(mean, std) * x_i + x_i
        x_j = augmentation_power * torch.normal(mean, std) * x_j + x_j

        return x_i, x_j

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def fit(self, x_train, epochs = 100, learning_rate = 0.0001,
            batch_size=256, backup_freq=5, warm_up_epochs=0, early_stopping_rounds=None,
            augmentation_power=0.2, instance_temperature=0.5, cluster_temperature=1.0, verbose_freq=10, save_best_in_ram=True):

        def train_epoch(train_data_loader, optimizer, verbose_freq):
            size = len(train_data_loader.dataset)
            self.encoder.train()

            i = 0
            epoch_start = datetime.now()
            verbose_step = (len( train_data_loader ) // verbose_freq) + 1
            corr_mask_batch_size = self.mask_correlated_clusters( train_data_loader.batch_size )
            corr_mask_clust_num = self.mask_correlated_clusters( self.cluster_num )
            corr_mask_batch_size = corr_mask_batch_size.to(self.device)
            corr_mask_clust_num = corr_mask_clust_num.to(self.device)
            for x in train_data_loader:

                # bug fix
                if len(x) == 1:
                    continue

                x = x.to(self.device)
                x_i, x_j = self.augment_batch( x, augmentation_power=augmentation_power )

                z_i, z_j, c_i, c_j = self.encoder(x_i, x_j)

                if z_i.shape[0] != train_data_loader.batch_size:
                    corr_mask_batch_size = self.mask_correlated_clusters(z_i.shape[0])
                    corr_mask_batch_size = corr_mask_batch_size.to(self.device)

                loss_instance = InstanceLoss(batch_size=z_i.shape[0], corr_mask_batch_size=corr_mask_batch_size.clone(),
                                             temperature=instance_temperature)(z_i, z_j)
                loss_cluster = ClusterLoss(class_num=self.cluster_num, corr_mask_clust_num=corr_mask_clust_num.clone(),
                                           temperature=cluster_temperature)(c_i, c_j)
                loss = loss_instance + loss_cluster
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i += 1
                if i % verbose_step == 0:
                    loss, current = loss.item(), i * len(x_i)
                    print("loss: {:.5}  [{}/{}] {}".format( loss, current, size, datetime.now() - epoch_start ))

                del x, x_i, x_j
                del z_i, z_j, c_i, c_j
                del loss_instance
                del loss_cluster
                del loss
                torch.cuda.empty_cache()

            epoch_end = datetime.now()
            print("Total epoch time: {}".format( epoch_end - epoch_start ))

        x_train_dataset = GreyClusterizerDataset(x_train)
        train_data_loader = DataLoader(x_train_dataset, batch_size=batch_size, shuffle=True)

        best_loss = np.inf
        no_loss_improve_count = 0
        best_clusterizer = None
        for i in range(epochs):
            print("Current model name: {}".format(self.model_name))
            print("Epoch: {}".format(i))

            if i < warm_up_epochs:
                optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=0.0000001, weight_decay=1e-2, betas=(0.9, 0.999))
            else:
                optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=learning_rate, weight_decay=1e-2, betas=(0.9, 0.999))

            train_epoch(train_data_loader, optimizer, verbose_freq)
            val_augmentation_power = augmentation_power / 10.0 + 0.01
            mean_loss = self.eval_clusterizer_(x_train, batch_size, instance_temperature, cluster_temperature, augmentation_power=val_augmentation_power)
            print("Mean validation loss: {}".format( mean_loss ))

            if early_stopping_rounds is not None:
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    print("New best loss: {}".format(best_loss))
                    no_loss_improve_count = 0
                    with torch.no_grad():
                        if save_best_in_ram or self.checkpoint_dir is None:
                            self.encoder = self.encoder.to("cpu")
                            best_clusterizer = deepcopy(self)
                            self.encoder.to( self.device )
                        else:
                            save(self, os.path.join(self.checkpoint_dir, self.model_name + ".pkl"))
                else:
                    no_loss_improve_count += 1
                    if no_loss_improve_count == early_stopping_rounds:
                        print("{} epochs loss didn't improve. Early stopping.".format( early_stopping_rounds ))
                        print("Best loss: {}".format(best_loss))
                        break
            else:
                if i % backup_freq == 0:
                    with torch.no_grad():
                        self.encoder.eval()
                        save(self, os.path.join(self.checkpoint_dir, self.model_name + "_{}.pkl".format(i)))

        if early_stopping_rounds is not None:
            print( "Fitting's ended." )
            print( "Loading (rollback) to the best model." )
            self.encoder.eval()
            self.encoder = self.encoder.cpu()
            del self.encoder
            gc.collect()
            if save_best_in_ram or self.checkpoint_dir is None:
                self.encoder = best_clusterizer.encoder.to( self.device )
                del best_clusterizer
            else:
                best_clusterizer = load(os.path.join(self.checkpoint_dir, self.model_name + ".pkl"))
                self.encoder = best_clusterizer.encoder

        del optimizer
        del train_data_loader
        self.encoder = self.encoder.to("cpu")
        torch.cuda.empty_cache()
        pass

    def eval_clusterizer_(self, x, batch_size, instance_temperature, cluster_temperature, augmentation_power=0.01):
        x_val = GreyClusterizerDataset(x)
        val_data_loader = DataLoader(x_val, batch_size=batch_size, shuffle=True)
        self.encoder.eval()

        corr_mask_batch_size = self.mask_correlated_clusters(val_data_loader.batch_size)
        corr_mask_clust_num = self.mask_correlated_clusters(self.cluster_num)
        corr_mask_batch_size = corr_mask_batch_size.to(self.device)
        corr_mask_clust_num = corr_mask_clust_num.to(self.device)
        sum_loss = 0.0
        batch_count = len( val_data_loader )
        for x in val_data_loader:
            x = x.to(self.device)
            x_i, x_j = self.augment_batch(x, augmentation_power=augmentation_power)

            z_i, z_j, c_i, c_j = self.encoder(x_i, x_j)

            if z_i.shape[0] != val_data_loader.batch_size:
                corr_mask_batch_size = self.mask_correlated_clusters(z_i.shape[0])
                corr_mask_batch_size = corr_mask_batch_size.to(self.device)

            loss_instance = InstanceLoss(batch_size=z_i.shape[0], corr_mask_batch_size=corr_mask_batch_size.clone(),
                                         temperature=instance_temperature)(z_i, z_j)
            loss_cluster = ClusterLoss(class_num=self.cluster_num, corr_mask_clust_num=corr_mask_clust_num.clone(),
                                       temperature=cluster_temperature)(c_i, c_j)
            loss = loss_instance + loss_cluster
            sum_loss += loss.item()
        mean_loss = sum_loss / batch_count

        return mean_loss


    def get_embeddings(self, x_test, batch_size=256):

        self.encoder = self.encoder.to(self.device)

        x_test = GreyClusterizerDataset(x_test)
        data_loader = DataLoader(x_test, batch_size=batch_size, shuffle=False)

        self.encoder.eval()

        y_pred = []
        with torch.no_grad():
            for x in tqdm(data_loader, desc="Making embeddings", colour="green"):
                x = x.to(self.device)
                pred = self.encoder.backbone_model(x)
                pred = torch.flatten( pred, 1 )

                pred = pred.to("cpu").detach().numpy()
                y_pred.append(pred)

        y_pred = np.vstack(y_pred)

        self.encoder = self.encoder.to("cpu")
        return y_pred

    def predict_batch_cluster_probas(self, x_test, batch_size=256, k_n=None):

        self.encoder = self.encoder.to(self.device)

        x_test = GreyClusterizerDataset(x_test)
        data_loader = DataLoader(x_test, batch_size=batch_size, shuffle=False)

        self.encoder.eval()

        y_pred = []
        with torch.no_grad():
            for x in tqdm(data_loader, desc="Making cluster probas", colour="green"):
                x = x.to(self.device)
                cluster_proba = self.encoder.get_cluster_proba(x)
                cluster_proba = torch.flatten( cluster_proba, 1 )

                cluster_proba = cluster_proba.to("cpu").detach().numpy()

                if k_n is not None:
                    cluster_proba = self.make_k_neighbour_probas(cluster_proba, k_n)

                y_pred.append(cluster_proba)

        y_pred = np.vstack(y_pred)

        self.encoder = self.encoder.to("cpu")

        return y_pred

    def predict_one_sample_cluster_proba(self, x, k_n=None):

        self.encoder = self.encoder.to(self.device)

        self.encoder.eval()

        with torch.no_grad():
            x = x.reshape((1, x.shape[0]))
            x = torch.Tensor( x )
            x = x.to(self.device)
            cluster_proba = self.encoder.get_cluster_proba(x)
            cluster_proba = torch.flatten( cluster_proba, 1 )
            cluster_proba = cluster_proba.to("cpu").detach().numpy()
            cluster_proba = cluster_proba.reshape( (-1, ) )

            if k_n is not None:
                cluster_proba = self.make_k_neighbour_probas( cluster_proba, k_n )

        self.encoder = self.encoder.to("cpu")

        return cluster_proba

    def make_k_neighbour_probas(self, cluster_probas, k_neighbours):

        if len(cluster_probas.shape) == 1:
            cluster_probas = cluster_probas.reshape( (1, -1) )

        kn_probas = []
        for i in range( len(cluster_probas) ):
            most_relevant_cluster_ids = np.argpartition(cluster_probas[i], -k_neighbours)[-k_neighbours:]
            relevant_probas = np.zeros( (cluster_probas[i].shape[0],) )
            relevant_probas[ most_relevant_cluster_ids ] = cluster_probas[i][ most_relevant_cluster_ids ]
            relevant_probas = relevant_probas / np.sum(relevant_probas)
            kn_probas.append( relevant_probas )
        kn_probas = np.array(kn_probas)

        if kn_probas.shape[0] == 1:
            kn_probas = kn_probas.reshape( (-1, ) )

        return kn_probas

    def predict_batch_cluster_labels(self, x_test, batch_size=256, k_n=None, verbose=True, save_gpu_memory=False):
        self.encoder = self.encoder.to(self.device)

        x_test = GreyClusterizerDataset(x_test)
        data_loader = DataLoader(x_test, batch_size=batch_size, shuffle=False)

        self.encoder.eval()

        y_pred = []
        with torch.no_grad():

            if verbose:
                data_batches = tqdm(data_loader, desc="Making cluster labels", colour="green")
            else:
                data_batches = data_loader

            for x in data_batches:
                x = x.to(self.device)
                pred = self.encoder.predict_cluster(x)

                pred = pred.to("cpu").detach().numpy()
                y_pred.append(pred)

        y_pred = np.hstack(y_pred)

        if save_gpu_memory:
            self.encoder = self.encoder.to("cpu")

        return y_pred

    def predict_one_sample_cluster_label(self, x, save_gpu_memory=False):
        self.encoder = self.encoder.to(self.device)

        self.encoder.eval()

        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.reshape((1, x.shape[0]))
            x = torch.Tensor( x )
            x = x.to(self.device)
            cluster_label = self.encoder.predict_cluster(x)
            cluster_label = cluster_label.to("cpu").detach().numpy()
            cluster_label = cluster_label[0]

        if save_gpu_memory:
            self.encoder = self.encoder.to("cpu")

        return cluster_label

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