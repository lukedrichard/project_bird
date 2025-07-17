import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchaudio #for audio augmentation
import random


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



#get data loaders
def get_dataloader(dataframe, batch_size, label_to_index, max_frames, augment):
    
    dataset = BirdSpectrogramDataset(dataframe, 
                                    base_path='/storage/courses/DSWorkflow_Copy/project_bird/flattened_npy_spectrograms/', 
                                    label_to_index=label_to_index,
                                    max_frames=max_frames,
                                    augment=augment)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=1, 
                            pin_memory=True, 
                            collate_fn=collate_fn,
                            drop_last=False)

    return dataloader