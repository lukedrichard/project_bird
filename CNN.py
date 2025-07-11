import numpy as np
import pandas as pd
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


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


#get metadata
metadata_df = pd.read_csv('metadata.csv')

#reconstruct train/val/test sets
train_df = metadata_df[metadata_df['split'] == 'train']
val_df = metadata_df[metadata_df['split'] == 'val']
test_df = metadata_df[metadata_df['split'] == 'test']


#Dataset class
class BirdSpectrogramDataset(Dataset):
    def __init__(self, dataframe, base_path, label_to_index, max_frames, augment=False):
        self.paths = dataframe['npy_filename'].values
        self.labels = dataframe['species'].values
        self.base_path = base_path
        self.label_to_index = label_to_index
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.base_path + self.paths[index]
        spectrogram = np.load(path).astype(np.float32)  # shape: [128, time]
        label = self.label_to_index[self.labels[index]]

        # Convert to tensor and add channel dim
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0) # shape: [1, 128, time]

        #augment the data
        width = spectrogram.shape[-1]
        mask_percent = 0.1
        if self.augment:
            time_mask_param = mask_percent * width
            freq_mask_param = mask_percent * 128

            time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
            freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

            spectrogram = freq_mask(spectrogram)
            spectrogram = time_mask(spectrogram)


        # Clip or pad
        time_dim = spectrogram.shape[-1]
        if time_dim > self.max_frames:
            spectrogram = spectrogram[:, :, :self.max_frames]
        elif time_dim < self.max_frames:
            pad_amount = self.max_frames - time_dim
            pad = torch.zeros((1, 128, pad_amount), dtype=torch.float32)
            spectrogram = torch.cat((spectrogram, pad), dim=-1)


        #normalize data
        mean = spectrogram.mean()
        std = spectrogram.std()
        if std > 0:
            spectrogram = (spectrogram - mean) / std
        else:
            spectrogram = spectrogram - mean  # Avoid divide-by-zero

        return spectrogram, label

#Custom collate function
chunk_size = 512
stride = 512

def chunk_collate_fn(batch, chunk_size=chunk_size, stride=stride):
    
    all_chunks = []
    all_labels = []

    for spectrogram, label in batch:
        time_length = spectrogram.shape[-1] #get length of spectrogram

        #check if spectrogrma is shorter than chunk size, so it is not excluded from trainin
        if time_length < chunk_size:
            # Pad the spectrogram on the time axis
            pad_width = chunk_size - time_length
            padded = F.pad(spectrogram, (0, pad_width))  # Pad last dim only
            all_chunks.append(padded)
            all_labels.append(label)
        else:
            #create overlapping chunks of chunk_size with overlap based on stride
            for start in range(0, time_length - chunk_size + 1, stride):
                end = start + chunk_size
                current_chunk = spectrogram[:,:, start:end]
                all_chunks.append(current_chunk)
                all_labels.append(label)

    if len(all_chunks) == 0:
        raise ValueError("No chunks created, Adjust chunk_size and stride")

    batched_chunks = torch.stack(all_chunks)
    labels = torch.tensor(all_labels)

    return batched_chunks, labels


#lookup dictionary for converting labels
label_to_index = {label: index for index, label in enumerate(sorted(metadata_df['species'].unique()))}


#Create the model architecture
class BirdCNN(nn.Module):
    
    def __init__(self, num_classes):
        super(BirdCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.AdaptiveAvgPool2d((1,1))
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.5),

            #nn.Linear(256,128),
            #nn.ReLU(),
            #nn.Dropout(0.4),

            #nn.Linear(128,64),
            #nn.ReLU(),
            #nn.Dropout(0.4),

            nn.Linear(256, num_classes),
            
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x




#model hyperparameters
learning_rate = 1e-4
num_epochs = 100

#dataset parameters
batch_size = 64
max_frames = 512 #max length of spectrograms: 256 ~ 3sec
augment = True #should training data be augmented


#get data loaders
train_data = BirdSpectrogramDataset(train_df, 
                                    base_path='/storage/courses/DSWorkflow_Copy/project_bird/flattened_npy_spectrograms/', 
                                    label_to_index=label_to_index,
                                    max_frames=max_frames,
                                    augment=augment)

train_loader = DataLoader(train_data, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=1, 
                          pin_memory=True, 
                          #collate_fn=chunk_collate_fn,
                          drop_last=False)


val_data = BirdSpectrogramDataset(val_df, 
                                  base_path='/storage/courses/DSWorkflow_Copy/project_bird/flattened_npy_spectrograms/', 
                                  label_to_index=label_to_index,
                                  max_frames=max_frames)

val_loader = DataLoader(val_data, 
                        batch_size=batch_size,
                        shuffle=True, 
                        num_workers=1, 
                        pin_memory=True, 
                        #collate_fn=chunk_collate_fn,
                        drop_last=False)


#instantiate model, initialize weights with xavier_uniform method
def initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

model = BirdCNN(num_classes = len(label_to_index))
model.apply(initialize_weights)  # Apply recursively to all layers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=1e-4) #!!!try weight_decay and amsgrad


#set the device used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#train the model
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()


            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()

    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = 100 * val_correct / val_total

    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")


#Visualize loss and accuracy
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.savefig('loss_plot.png')  # Save loss plot
plt.close()  # Close the figure so it doesn't display


plt.figure()
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Over Epochs')
plt.savefig('accuracy_plot.png')  # Save accuracy plot
plt.close()  # Close the figure

# Save the model
torch.save(model, 'bird_cnn.pth')
