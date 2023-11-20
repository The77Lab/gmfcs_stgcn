
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DataSet_Classification, DataSet_Classification_Augmented
import math
import os
import sys
# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from pytorch_metric_learning import losses, miners

from network import STGCN_Classifier_Metric
batch_size = 96
sample_folder = "C:/Users/pjzha/OneDrive/Desktop/video-gait-v1/samples/"

for sample_rate in range(1, 2):
    ratio = sample_rate / 10

    train_dataset = DataSet_Classification_Augmented(os.path.join(script_dir, 'sampled_trainingset/sampled_' + str(ratio) + '_' +str(0) + '.npy'),
                            sample_folder, data_augmentation=False)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataset2 = DataSet_Classification(os.path.join(script_dir, 'sampled_trainingset/sampled_' + str(ratio) + '_' +str(0) + '.npy'),
                            sample_folder, data_augmentation=False)
    train_dataloader2 = DataLoader(
        train_dataset2, batch_size=batch_size, shuffle=True)
    val_dataset = DataSet_Classification(os.path.join(script_dir, 'val_dataset14.npy'),
                        sample_folder, data_augmentation=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model_weights_path = 'sampled_models_triplet/model_' + str(ratio) + '_' + str(0) + '.pth'

    backbone_cfg = {
        'type': 'STGCN',
        'gcn_adaptive': 'init',
        'gcn_with_res': True,
        'tcn_type': 'mstcn',
        'graph_cfg': {
            'layout': 'coco',
            'mode': 'spatial'
        },
        'pretrained': None
    }
    model = STGCN_Classifier_Metric(backbone=backbone_cfg, num_classes=4)

    device = 'cuda:0'


    # Load pre-trained weights to the backbone
    backbone_state_dict = os.path.join(script_dir, 'j.pth')
    # load_checkpoint(model.backbone, backbone_state_dict)
    tmp = torch.load(backbone_state_dict)
    # print(tmp.keys())

    del tmp['cls_head.fc_cls.weight']
    del tmp['cls_head.fc_cls.bias']
    # print(tmp.keys())
    model.load_state_dict(tmp, strict=False)

    # print(model)

    for param in model.backbone.parameters():
        param.requires_grad = False

    model = model.to(device)

    triplet_loss = losses.TripletMarginLoss(margin=0.6)
    miner = miners.TripletMarginMiner(margin=0.6, type_of_triplets="all")

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)

    val_best_acc = -math.inf

    num_epochs = 40
    for epoch in range(num_epochs):
        if epoch == 0:
            for layer in range(9, 3, -1):
                for param in model.backbone.gcn[layer].parameters():
                    param.requires_grad = True
        train_loss = 0.0
        train_correct = 0
        model.train()  # set the model to train mode
        epoch_pbar = tqdm(desc=f"Epoch {epoch+1}/{num_epochs}",
                        total=len(train_dataloader.dataset) / batch_size, position=0)

        epoch_acc = 0.0
        epoch_loss = 0.0

        sample_count = 0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            optimizer.zero_grad()
            inputs = inputs.view(-1, 1, 124, 17 ,3)
            # Forward pass
            outputs = model(inputs)
            # p_labels
            # positive_pairs = (torch.tensor(range(0,outputs.shape[0],2)), torch.tensor(range(1,outputs.shape[0],2)), torch.Tensor([]), torch.Tensor([]))
            
            # contrastive_loss = self_supervised_loss(outputs, p_labels, positive_pairs)
            # print(outputs.shape)
            outputs = outputs.view(-1, 2, 256)
            # outputs = outputs.view(-1, 3, 512)
            outputs = outputs.transpose(0, 1)
            # print(outputs.shape)
            pairwise_dist = F.pairwise_distance(F.normalize(outputs[0]), F.normalize(outputs[1]))
            contrastive_loss = torch.mean(pairwise_dist)

            hard_pairs_1 = miner(outputs[0], labels)

            
            triplet_loss1 = triplet_loss(outputs[0], labels, hard_pairs_1)
            loss = triplet_loss1 + contrastive_loss

            loss.backward()
            optimizer.step()
            sample_count += batch_size


            # we cannot get label when using triplet loss

            epoch_loss += loss.item()
            tmp_loss = epoch_loss * 1.0 / sample_count * batch_size

            # Update the progress bar for the epoch
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({'loss': tmp_loss})

        # Compute the training loss and accuracy for this epoch
        epoch_loss /= (len(train_dataloader.dataset) / batch_size)
        # Close the progress bar for the epoch
        epoch_pbar.close()
        print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.4f}')


        # Evaluate the model on the validation set
        val_loss = 0.0
        val_correct = 0
        model.eval()  # set the model to eval mode

        # we first compute and save the embeddings of the training data
        
        print('computing the embeddings of the training data')
        
        train_embeddings = np.array([])
        train_labels = np.array([])
        with torch.no_grad():
            for inputs, labels in train_dataloader2:
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                # Forward pass
                outputs = model(inputs)
                # append the embeddings and labels
                if train_embeddings.size == 0:
                    train_embeddings = outputs.cpu().detach().numpy()
                    train_labels = labels.cpu().detach().numpy()
                else:
                    train_embeddings = np.concatenate((train_embeddings, outputs.cpu().detach().numpy()), axis=0)
                    train_labels = np.concatenate((train_labels, labels.cpu().detach().numpy()), axis=0)
        print(train_embeddings.shape)
        # for each validation sample we compute the distance to the training samples
        # and find the closest one

        epoch_pbar = tqdm(desc=f"VAL Epoch {epoch+1}/{num_epochs}",
                        total=len(val_dataloader.dataset) / batch_size, position=0)
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                # Forward pass
                outputs = model(inputs)

                outputs = outputs.cpu().detach().numpy()

                # for each sample in the batch we compute the distance to the training samples
                # and find the closest one
                for i in range(outputs.shape[0]):
                    distances = np.linalg.norm(train_embeddings - outputs[i][np.newaxis, :], axis=1)
                    if train_labels[np.argmin(distances)] == labels[i]:
                        val_correct += 1

                epoch_pbar.update(1)
        epoch_pbar.close()

        # Compute the validation loss and accuracy for this epoch
        val_accuracy = 100.0 * val_correct / len(val_dataloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs} -  Validation Accuracy: {val_accuracy:.2f}%')

        if val_accuracy > val_best_acc:
            val_best_acc = val_accuracy
            torch.save(model.state_dict(), os.path.join(script_dir, model_weights_path))
            print('model saved!')

