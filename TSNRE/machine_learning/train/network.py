import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)


from models.heads import Classifier, Classifier_Metric
from models.stgcn import STGCN

class STGCN_Classifier(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes=0):
        super(STGCN_Classifier, self).__init__()

        args = backbone.copy()
        args.pop('type')
        self.backbone = STGCN(**args)
        self.cls_head = Classifier(
            num_classes=num_classes, dropout=0.5, latent_dim=512)

    def forward(self, keypoint):
        """Define the computation performed at every call."""
        x = self.backbone(keypoint)
        cls_score = self.cls_head(x)
        return cls_score
    
    

class STGCN_Classifier_Metric(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes=0):
        super(STGCN_Classifier_Metric, self).__init__()

        args = backbone.copy()
        args.pop('type')
        self.backbone = STGCN(**args)
        # self.cls_head = Classifier_Metric(
        #     num_classes=num_classes, dropout=0.5, latent_dim=512)
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling = nn.AvgPool2d((31, 17))

    def forward(self, x):
        """Define the computation performed at every call."""
        x = self.backbone(x)
        x = x.squeeze(1)
        # print(x.shape)
        # cls_score = self.cls_head(x)
        x = self.pooling(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)

        return x