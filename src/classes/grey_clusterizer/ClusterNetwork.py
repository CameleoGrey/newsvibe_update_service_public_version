


import torch
import torch.nn as nn
from torch.nn.functional import normalize
from src.classes.grey_clusterizer.MLP_Network import MLP_Network


class ClusterNetwork(nn.Module):
    def __init__(self, input_dim, latent_size=100, hidden_layer_dim=512, hidden_layers_num=10,
                    dropout_rate=0.05, cluster_num=100, instance_feature_dim=100, device="cuda"):
        super(ClusterNetwork, self).__init__()

        self.feature_dim = instance_feature_dim
        self.cluster_num = cluster_num
        self.device = device

        self.backbone_model = MLP_Network( input_dim=input_dim, output_dim=latent_size,
                                           hidden_layer_dim=hidden_layer_dim, hidden_layers_num=hidden_layers_num,
                                           dropout_rate=dropout_rate)

        self.instance_projector = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, instance_feature_dim)
        )

        self.cluster_projector = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cluster_num),
            nn.Softmax(dim=1)
        )

        self.backbone_model = self.backbone_model.to(self.device)
        self.instance_projector = self.instance_projector.to(self.device)
        self.cluster_projector = self.cluster_projector.to(self.device)

    def forward(self, x_i, x_j):
        h_i = self.backbone_model(x_i)
        h_j = self.backbone_model(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def predict_cluster(self, x):
        h = self.backbone_model(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

    def get_cluster_proba(self, x):
        h = self.backbone_model(x)
        c = self.cluster_projector(h)
        return c
