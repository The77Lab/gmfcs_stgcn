import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def normal_init(module: nn.Module,
                mean: float = 0,
                std: float = 1,
                bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class Classifier(nn.Module):
    def __init__(self, num_classes=4, latent_dim=512, in_channels=256, dropout=0.4, init_std=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.out = num_classes
        self.latent_dim = latent_dim
        
        self.linear1 = nn.Linear(self.in_channels, self.latent_dim)
        self.fc_cls = nn.Linear(self.latent_dim, self.out)
        
        self.pool = nn.MaxPool2d((31, 17))
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        normal_init(self.linear1, std=init_std)
        normal_init(self.fc_cls, std=init_std)
    
    def forward(self, x):

        # pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N, C, T, V)
        x = self.pool(x)
        x = x.reshape(N, C)

        assert x.shape[1] == self.in_channels
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        cls_score = self.fc_cls(x)

        return cls_score



class Classifier_Metric(nn.Module):
    def __init__(self, num_classes=4, latent_dim=512, in_channels=256, dropout=0.4, init_std=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.out = num_classes
        self.latent_dim = latent_dim
        
        self.linear1 = nn.Linear(self.in_channels, self.latent_dim)
        self.fc_cls = nn.Linear(self.latent_dim, self.out)
        
        self.pool = nn.MaxPool2d((31, 17))
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        normal_init(self.linear1, std=init_std)
        normal_init(self.fc_cls, std=init_std)
    
    def forward(self, x):

        # pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N, C, T, V)
        x = self.pool(x)
        x = x.reshape(N, C)

        assert x.shape[1] == self.in_channels
        x = self.linear1(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        # cls_score = self.fc_cls(x)

        return x