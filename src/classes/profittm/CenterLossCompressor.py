import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.classes.utils import save, load
from torch.utils.data import DataLoader, Dataset


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(
            batch_size if self.size_average else 1)
        loss = self.centerlossfunc(
            feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(
            0, label.unsqueeze(1).expand(
                feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


class CenterLossNN(nn.Module):
    def __init__(self, x_shape, n_classes, latent_dim):
        super(CenterLossNN, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=2)
        torch.nn.init.kaiming_normal_(self.conv1_1.weight)
        self.bn_1 = nn.BatchNorm2d(32)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=2)
        torch.nn.init.kaiming_normal_(self.conv1_2.weight)
        self.bn_2 = nn.BatchNorm2d(64)
        self.do_1 = nn.Dropout2d(p=0.2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=2)
        torch.nn.init.kaiming_normal_(self.conv2_1.weight)
        self.bn_3 = nn.BatchNorm2d(128)
        self.prelu3_2 = nn.PReLU()
        
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128 * (x_shape[2] - 3), latent_dim)
        self.ip2 = nn.Linear(latent_dim, n_classes, bias=True)

    def forward(self, x):
        x = self.prelu1_1(self.bn_1(self.conv1_1(x)))
        x = self.prelu1_2(self.do_1(self.bn_2(self.conv1_2(x))))
        x = self.prelu3_2(self.bn_3(self.conv2_1(x)))
        
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        ip1 = self.preluip1(self.ip1(x))
        
        ip2 = self.ip2(ip1)
        ip2 = F.log_softmax(ip2, dim=1)
        
        return ip1, ip2

class MLP_Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dim, hidden_layers_num=2, dropout_rate=0.2):
        super( MLP_Network, self).__init__()
        
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_dim),
            nn.BatchNorm1d(hidden_layer_dim),
            nn.PReLU()
        )
        self.input_layer.apply(init_weights)
        

        self.middle_layers = []
        for i in range( hidden_layers_num ):
            hidden_layer = nn.Sequential(
                nn.Linear(hidden_layer_dim, hidden_layer_dim),
                nn.BatchNorm1d(hidden_layer_dim),
                nn.Dropout(p=dropout_rate),
                nn.PReLU(),
            )
            hidden_layer.apply(init_weights)
            self.middle_layers.append( hidden_layer )
        self.middle_layers = nn.Sequential( * self.middle_layers )
        
        self.latent_output = nn.Sequential(
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            #nn.BatchNorm1d(hidden_layer_dim),
            nn.PReLU(),
        )
        self.latent_output.apply(init_weights)
        
        self.classes_output = nn.Linear(hidden_layer_dim, output_dim, bias=True)
        nn.init.kaiming_uniform_(self.classes_output.weight, mode='fan_in', nonlinearity='relu')

        pass

    def forward(self, x):
        x = self.input_layer(x)
        x = self.middle_layers(x)
        
        latent_features = self.latent_output(x)
        
        classes_output = self.classes_output(latent_features)
        classes_output = F.log_softmax(classes_output, dim=1)
        
        
        return latent_features, classes_output

class CompressorTrainDataset(Dataset):
    def __init__(self, x, y, device, stack_count):
        
        self.x = x
        self.y = y
        self.device = device
        self.stack_count = stack_count
        
        pass
    
    def __getitem__(self, id):
        
        if self.stack_count > 1:
            stacked_x = np.vstack([self.x[id] for i in range(self.stack_count)])
            stacked_x = torch.Tensor(stacked_x)
            stacked_x = torch.unsqueeze(stacked_x, dim=0)
        else:
            stacked_x = self.x[id]
            stacked_x = torch.Tensor(stacked_x)

        stacked_x = stacked_x.to(self.device)
        
        y = self.y[id]
        y = torch.tensor(y, dtype=torch.int64)
        y = y.to(self.device)
        
        return stacked_x, y
    
    def __len__(self):
        
        data_length = len(self.y)
        
        return data_length

class CompressorTestDataset(Dataset):
    def __init__(self, x, device, stack_count):
        
        self.x = x
        self.device = device
        self.stack_count = stack_count
        
        pass
    
    def __getitem__(self, id):
        
        if self.stack_count > 1:
            stacked_x = np.vstack([self.x[id] for i in range(self.stack_count)])
            stacked_x = torch.Tensor(stacked_x)
            stacked_x = torch.unsqueeze(stacked_x, dim=0)
        else:
            stacked_x = self.x[id]
            stacked_x = torch.Tensor(stacked_x)
        
        stacked_x = stacked_x.to(self.device)
        
        return stacked_x
    
    def __len__(self):
        
        data_length = len(self.x)
        
        return data_length

class CenterLossCompressor():
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda')

        self.latent_dim = None
        self.n_classes = None
        self.stack_count = None

        pass

    def fit(self, x, y, validation_part=0.05,
            batch_size=100, epochs=100, latent_dim=100, stack_count=1):
        
        self.n_classes = len(np.unique(y))
        self.latent_dim = latent_dim
        self.stack_count = stack_count

        """self.model = CenterLossNN(
            x_shape=(len(x), self.stack_count, len(x[0])),
            n_classes=self.n_classes,
            latent_dim=latent_dim)"""
        
        self.model = MLP_Network(input_dim=len(x[0]), output_dim=self.n_classes, 
                                 hidden_layer_dim=latent_dim, 
                                 hidden_layers_num=2, 
                                 dropout_rate=0.05)
        
        self.model.to(self.device)
        
        self.model.train()

        loss_weight = 1
        nllloss = nn.CrossEntropyLoss().to(self.device)
        centerloss = CenterLoss(self.n_classes, self.latent_dim).to(self.device)
        optimizer4nn = optim.Adam(self.model.parameters(), lr=0.001)
        optimzer4center = optim.Adam(centerloss.parameters(), lr=0.5)
        
        train_dataset = CompressorTrainDataset(x, y, device="cuda", stack_count=stack_count)
        
        compressor_batch_size = len(x) // 100 + 2
        if compressor_batch_size > 256:
            compressor_batch_size = 256
        print("Compressor batch size: {}".format(compressor_batch_size))
        train_dataloader = DataLoader(train_dataset, batch_size=compressor_batch_size, shuffle=True, drop_last=True)

        for epoch in range(epochs):
            for batch_x, batch_y in tqdm(train_dataloader, desc='CenterLossCompressor fit | Epoch {} of {}'.format(epoch+1, epochs)):
                ip1, pred = self.model(batch_x)
                loss = nllloss(pred, batch_y) + loss_weight * centerloss(batch_y, ip1)

                optimizer4nn.zero_grad()
                optimzer4center.zero_grad()
                loss.backward()
                optimizer4nn.step()
                optimzer4center.step()
        torch.cuda.empty_cache()
        pass

    def predict(self, x, batch_size=100):
        
        self.model.eval()
        
        test_dataset = CompressorTestDataset(x, device="cuda", stack_count=self.stack_count)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        features = []
        for x in test_dataloader:
            with torch.no_grad():
                feats, labels = self.model(x)
                feats = feats.data.cpu().numpy()
                features.append(feats)
                batch = x.to('cpu')
                del batch
        features = np.vstack(features)
        torch.cuda.empty_cache()

        return features
