
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import sys

# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)
import numpy as np

from models.heads import Classifier
from models.stgcn import STGCN
from utils.data_processing import Preprocess_Module


transform = Preprocess_Module(data_augmentation=False)


class STGCN_Classifier(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes=0):
        super(STGCN_Classifier, self).__init__()

        args = backbone.copy()
        args.pop('type')
        self.backbone = STGCN(**args)
        self.cls_head = Classifier(
            num_classes=num_classes, in_channels=256, dropout=0.5, latent_dim=512)

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone.init_weights()
        self.cls_head.init_weights()

    def forward(self, keypoint):
        """Define the computation performed at every call."""
        x = self.backbone(keypoint)
        # print(x)
        # print(x)
        cls_score = self.cls_head(x)
        # print(cls_score)
        # cls_score = F.softmax(cls_score, dim=1)
        return cls_score



sample_folder = 'samples dir'

backbone_cfg = {
    'type': 'STGCN',
    'gcn_adaptive': 'init',
    'gcn_with_res': True,
    'tcn_type': 'mstcn',
    'num_stages': 10,
    'inflate_stages': [5, 8],
    'down_stages': [5, 8],
    'graph_cfg': {
        'layout': 'coco',
        'mode': 'spatial'
    },
    'pretrained': None
}

model = STGCN_Classifier(backbone=backbone_cfg, num_classes=4)

device = 'cuda:0'


# Load pre-trained weights to the backbone
state_dict = os.path.join(script_dir, 'model_weights.pth')
# load_checkpoint(model.backbone, backbone_state_dict)
tmp = torch.load(state_dict)
model.load_state_dict(tmp, strict=True)

model = model.to(device)
model.eval()

def load_sample(video, clip):
    data_file_path = os.path.join(sample_folder, str(video), str(video) + '_' + str(clip) + '.npy')
    data = np.load(data_file_path)
    tmp_dict = {}
    tmp_dict['img_shape'] = (320, 480)
    tmp_dict['label'] = -1
    tmp_dict['start_index'] = 0
    tmp_dict['modality'] = 'Pose'
    tmp_dict['total_frames'] = 124
    data = np.where(data == 0, 1e-4, data)
    data[np.isnan(data)] = 1e-4
    tmp_dict['keypoint'] = data[np.newaxis, :, :, :2]
    tmp_dict['keypoint'] = np.tile(tmp_dict['keypoint'], (2, 1, 1, 1))
    tmp_dict['keypoint_score'] = data[np.newaxis, :, :, 2] #because we do not have class 0
    tmp_dict['keypoint_score'] = np.tile(tmp_dict['keypoint_score'], (2, 1, 1))
    
    data = transform(tmp_dict)
    data = data['keypoint'][0]

    data = data.numpy()
    return data


test_dataset_file = os.path.join(script_dir, 'test_dataset14.npy')

test_data = np.load(test_dataset_file)


test_data_dict = {}
for i in range(len(test_data)):
    label, _, video, clip = test_data[i]
    if video not in test_data_dict:
        test_data_dict[video] = {}
        test_data_dict[video]['label'] = label - 1
        test_data_dict[video]['clip'] = []
    test_data_dict[video]['clip'].append(clip)

    
total_correct = 0
total_count = 0
correct_matrix = np.zeros((4, 4))
for video in tqdm(test_data_dict):
    label = test_data_dict[video]['label']
    test_data = []
    for clip in test_data_dict[video]['clip']:
        test_data.append(load_sample(video, clip))
    test_data = np.array(test_data)
    test_data = torch.from_numpy(test_data).float().to(device)
    
    sample_result = model(test_data)
    sample_result = sample_result.cpu().detach().numpy()
    sample_result = np.argmax(sample_result, axis=1)
    sample_result = np.bincount(sample_result)
    sample_result = np.argmax(sample_result)
    if sample_result == label:
        total_correct += 1
    total_count += 1
    correct_matrix[sample_result][label] += 1
print('accuracy: ', total_correct / total_count)
print(correct_matrix)
