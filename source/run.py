import numpy as np
import pandas as pd
import random
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchaudio #for audio augmentation

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import get_dataloader, get_chunk_dataloader
from cnn_architecture import get_BaseBird
from trainer import train_model, train_chunk_model
from evaluator import evaluate, chunk_evaluate

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
#set the device used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set paths
#metadata_path = '../data/metadata/metadata.csv'
#base_path = '../data/processed_audio/flattened_npy_spectrograms/' #directory where data is stored
#results_dir = '../results/test_run/'
#noise_dir = "../data/processed_audio/128f_chunked_spectrograms/noise"
#noise_paths = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.npy')]

#paths for chunked spectrograms
chunk_metadata_path = "../data/metadata/128f_chunk_metadata.csv"
base_path = '' #don't need metadata has full paths
results_dir = '../results/final_chunk_model/'
noise_dir = "../data/processed_audio/128f_chunked_spectrograms/noise"
noise_paths = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.npy')]
chunk_base_path = '../data/processed_audio/128f_chunked_spectrograms'

#make directory for results
os.makedirs(results_dir, exist_ok=True)

#get metadata
#metadata_df = pd.read_csv(metadata_path)
#for chunked metadata
chunk_metadata_df = pd.read_csv(chunk_metadata_path)
chunk_metadata_df = chunk_metadata_df[chunk_metadata_df['species'] != 'noise']
chunk_metadata_df['original_file'] = chunk_metadata_df['original_file'].str.replace('.npy', '', regex=False)
mix_paths = chunk_metadata_df['chunk_path'].values

#reconstruct train/val/test sets
#train_df = metadata_df[metadata_df['split'] == 'train']
#val_df = metadata_df[metadata_df['split'] == 'val']
#test_df = metadata_df[metadata_df['split'] == 'test']


#reconstruct train/val/test sets
chunk_train_df = chunk_metadata_df[chunk_metadata_df['split'] == 'train']
chunk_val_df = chunk_metadata_df[chunk_metadata_df['split'] == 'val']
chunk_test_df = chunk_metadata_df[chunk_metadata_df['split'] == 'test']

eval_train_df = chunk_train_df[['original_file', 'species']].drop_duplicates().reset_index(drop=True)
eval_val_df = chunk_val_df[['original_file', 'species']].drop_duplicates().reset_index(drop=True)
eval_test_df = chunk_test_df[['original_file', 'species']].drop_duplicates().reset_index(drop=True)

#lookup dictionary for converting labels
label_to_index = {label: index for index, label in enumerate(sorted(chunk_metadata_df['species'].unique()))}
#label_to_index['noise'] = len(label_to_index) 

#model hyperparameters
learning_rate = 1e-4
num_epochs = 100
num_classes = len(label_to_index)

#dataset parameters
batch_size = 1
max_frames = 512 #max length of spectrograms: 256 ~ 3sec
#max_frames = 128 #all chunks are 128 frames ~ 1.5sec
augment = True #should training data be augmented

#train_loader = get_dataloader(chunk_train_df, base_path, noise_paths, batch_size, label_to_index, max_frames, augment)
#val_loader = get_dataloader(chunk_val_df, base_path, noise_paths, batch_size, label_to_index, max_frames, augment=False)

train_loader = get_chunk_dataloader(eval_train_df, chunk_base_path, noise_paths, mix_paths, label_to_index, batch_size=1, augment=True)
val_loader = get_chunk_dataloader(eval_val_df, chunk_base_path, noise_paths, mix_paths, label_to_index, batch_size=1, augment=False)


model = get_BaseBird(num_classes = len(label_to_index)).to(device)
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #!!!try weight_decay and amsgrad

train_chunk_model(model, num_epochs, train_loader, val_loader, device, optimizer, criterion, num_classes, results_dir)


#get evaluation metrics
#train_loader = get_dataloader(chunk_train_df, base_path, noise_paths, batch_size, label_to_index, max_frames, augment=False)
#evaluate(device, model, data_loader=train_loader, results_dir=results_dir + 'train_', label_to_index=label_to_index)
#evaluate(device, model, data_loader=val_loader, results_dir=results_dir + 'val_', label_to_index=label_to_index)


#If doing final test evaluation
#test_loader = get_dataloader(test_df, base_path, noise_paths, batch_size, label_to_index, max_frames, augment=False)
#evaluate(device, model, data_loader=test_loader, results_dir=results_dir + 'test_', label_to_index=label_to_index)

# Save the model
#torch.save(model, results_dir + 'BaseBird.pth')

#get train loader without augmentation/ change to original metadata on chunked/!!!set batch to 1
train_loader = get_chunk_dataloader(eval_train_df, chunk_base_path, noise_paths, mix_paths, label_to_index, batch_size=1, augment=False)
chunk_evaluate(device, model, data_loader=train_loader, results_dir=results_dir + 'train_', label_to_index=label_to_index)

val_loader = get_chunk_dataloader(eval_val_df, chunk_base_path, noise_paths, mix_paths, label_to_index, batch_size=1, augment=False)
chunk_evaluate(device, model, data_loader=val_loader, results_dir=results_dir + 'val_', label_to_index=label_to_index)


# Save the model
torch.save(model, results_dir + 'ChunkyBaseBird.pth')

# !!! for final testing only
test_loader = get_chunk_dataloader(eval_test_df, chunk_base_path, noise_paths, mix_paths, label_to_index, batch_size=1, augment=False)
chunk_evaluate(device, model, data_loader=test_loader, results_dir=results_dir + 'test_', label_to_index=label_to_index)
