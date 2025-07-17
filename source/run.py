import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchaudio #for audio augmentation

import cv2

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import get_dataloader, collate_fn
from cnn_architecture import get_BirdCNN
from trainer import train_model


#set the device used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get metadata
metadata_df = pd.read_csv('metadata.csv')

#reconstruct train/val/test sets
train_df = metadata_df[metadata_df['split'] == 'train']
val_df = metadata_df[metadata_df['split'] == 'val']
test_df = metadata_df[metadata_df['split'] == 'test']

#lookup dictionary for converting labels
label_to_index = {label: index for index, label in enumerate(sorted(metadata_df['species'].unique()))}
label_to_index['noise'] = len(label_to_index) 

#directory where data is stored
base_path = '/home/ldrich/Summer2025BHT/deep_learning/data_project_bird/flattened_npy_spectrograms/'
#base_path = '/storage/courses/DSWorkflow_Copy/project_bird/flattened_npy_spectrograms/'

#model hyperparameters
learning_rate = 1e-4
num_epochs = 100
num_classes = len(label_to_index) #be careful about whether noise category was added or not

#dataset parameters
batch_size = 64
max_frames = 512 #max length of spectrograms: 256 ~ 3sec
augment = True #should training data be augmented

#collate function parameters
chunk_size = 256
stride = 64
snr_threshold = 0.0001
noise_label = label_to_index['noise']

#collate object, need to pass it's function to dataloader
custom_collate_fn = collate_fn(chunk_size, stride, snr_threshold, noise_label)

train_loader = get_dataloader(train_df, batch_size, label_to_index, max_frames, augment)
val_loader = get_dataloader(val_df, batch_size, label_to_index, max_frames, augment)

#instantiate model, initialize weights with xavier_uniform method
def initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

model = get_BirdCNN(num_classes = len(label_to_index)).to(device)
model.apply(initialize_weights)  # Apply recursively to all layers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #!!!try weight_decay and amsgrad


train_model(model, num_epochs, train_loader, val_loader, device, optimizer, criterion, num_classes)
