import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchaudio #for audio augmentation
import random
from pathlib import Path
import os


#Dataset class
class BirdSpectrogramDataset(Dataset):
    def __init__(self, dataframe, base_path,  noise_paths, label_to_index, max_frames, augment=False):
        #self.paths = dataframe['npy_filename'].values
        self.paths = dataframe['chunk_path'].values
        self.labels = dataframe['species'].values
        self.base_path = base_path
        self.noise_paths = noise_paths
        self.label_to_index = label_to_index
        self.max_frames = max_frames
        self.augment = augment
        self.num_classes = len(label_to_index)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.base_path + self.paths[index]
        spectrogram = np.load(path).astype(np.float32)  # shape: [128, time]

        #both hard and soft labels are returned. Soft labels will be needed for mix up augmentation
        label_idx = self.label_to_index[self.labels[index]]
        label_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        label_onehot[label_idx] = 1.0

        #clip spectrogram to max_frames
        #spectrogram = self.clip(spectrogram)

        # data augmentation logic
        if self.augment:
            #apply mix-up
            if random.random() < 0.5:
                spectrogram, label_onehot = self.mixup(spectrogram, label_onehot, alpha=1.0)
            # add background noise
            if random.random() < 0.5:
                spectrogram = self.inject_background_noise(spectrogram, blend_factor=0.5)

            #add gaussian noise
            if random.random() < 0.5:
                spectrogram = self.inject_gaussian_noise(spectrogram, mean=0, min_std=0.05, max_std=0.05)

            #apply vertical roll
            if random.random() < 0.5:
                spectrogram = self.vertical_roll(spectrogram, vertical=0.05)

            #apply gain
            if random.random() < 0.5:
                spectrogram = self.apply_gain(spectrogram, min_gain_db=-12.0, max_gain_db=12.0)

        # Convert to tensor and add channel dim
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0) # shape: [1, 128, time]


        #augment the data
        #width = spectrogram.shape[-1]
        #mask_percent = 0.2
        #if self.augment and random.random() < 0.5:
        #    time_mask_param = mask_percent * width
        #    freq_mask_param = mask_percent * 128

        #    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        #    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

        #    spectrogram = freq_mask(spectrogram)
        #    spectrogram = time_mask(spectrogram)


        #normalize data
        mean = spectrogram.mean()
        std = spectrogram.std()
        if std > 0:
            spectrogram = (spectrogram - mean) / std
        else:
            spectrogram = spectrogram - mean  # Avoid divide-by-zero

        return spectrogram, label_idx, label_onehot

    
    def clip(self, spectrogram):
        """Clip time dimension of original spectrogram to max_frames. Pad if spectrogram is shorter than max_frames"""
        n_mels, time_dim = spectrogram.shape
        if time_dim > self.max_frames:
            spectrogram = spectrogram[:, :self.max_frames]
        elif time_dim < self.max_frames:
            pad_width = self.max_frames - time_dim
            spectrogram = np.pad(spectrogram, ((0,0), (0, pad_width)), mode='constant')

        return spectrogram

    def mixup(self, spectrogram, label_onehot, alpha=1.0):
        mix_idx = np.random.randint(0,len(self)) #get random spectrogram from batch
        mix_path = self.base_path + self.paths[mix_idx]
        mix_spec = np.load(mix_path).astype(np.float32)  # shape: [128, time]
        mix_spec = self.clip(mix_spec) # clip since this is original spectrogram

        #make one hot label for the mixup spectrogram
        mix_label_idx = self.label_to_index[self.labels[mix_idx]]
        mix_label_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        mix_label_onehot[mix_label_idx] = 1.0

        lam = np.random.beta(alpha, alpha) #mixup ratio
        spectrogram = lam*spectrogram + (1-lam)*mix_spec
        new_label = lam*label_onehot + (1-lam)*mix_label_onehot

        return spectrogram, new_label


    def inject_background_noise(self, spectrogram, blend_factor):
        noise = np.load(random.choice(self.noise_paths))  # shape: should match spec
        if noise.shape != spectrogram.shape:
            noise = self._resize_noise(noise, spectrogram.shape)
        #add the noise and return
        return spectrogram + (blend_factor * noise)

    def _resize_noise(self, noise, target_shape):
        """Resize noise to match the target spectrogram shape."""
        n_mels, time = target_shape
        noise_mels, noise_time = noise.shape

        # Time dimension
        if noise_time < time:
            repeats = (time // noise_time) + 1
            noise = np.tile(noise, (1, repeats))[:, :time]
        elif noise_time > time:
            noise = noise[:, :time]

        # Mel dimension
        if noise_mels < n_mels:
            noise = np.pad(noise, ((0, n_mels - noise_mels), (0, 0)))
        elif noise_mels > n_mels:
            noise = noise[:n_mels, :]

        return noise

    def inject_gaussian_noise(self, spectrogram, mean=0, min_std=0.001, max_std=0.015):
        std = np.random.uniform(min_std, max_std)
        noise = np.random.normal(mean, std, size=spectrogram.shape).astype(np.float32)
        return spectrogram + noise
    
    def vertical_roll(self, spectrogram, vertical=0.1):
        v_shift = int(spectrogram.shape[0] * random.uniform(-vertical, vertical))
        spectrogram = np.roll(spectrogram, v_shift, axis=0)
        return spectrogram


    def apply_gain(self, spectrogram, min_gain_db=-12.0, max_gain_db=12.0):
        gain_db = random.uniform(min_gain_db, max_gain_db)
        gain_linear = 10 ** (gain_db / 20)
        return spectrogram * gain_linear




class ChunkSpectrogramDataset(Dataset):
    def __init__(self, dataframe, base_path,  noise_paths, mix_paths, label_to_index, augment=False):
        #self.paths = dataframe['npy_filename'].values
        self.chunk_dirs = dataframe['original_file'].values
        self.labels = dataframe['species'].values
        self.base_path = base_path
        self.noise_paths = noise_paths
        self.mix_paths = mix_paths
        self.label_to_index = label_to_index
        #self.max_frames = max_frames
        self.augment = augment
        self.num_classes = len(label_to_index)

    def __len__(self):
        return len(self.chunk_dirs)

    def __getitem__(self, idx):
        dir_path = os.path.join(self.base_path, self.chunk_dirs[idx])

        # Single label per recording
        label_idx = self.label_to_index[self.labels[idx]]
        label_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        label_onehot[label_idx] = 1.0

        #get all of the chunks; apply augmentation; convert to tensors
        chunk_files = sorted(list(Path(dir_path).glob("*.npy")))
        inputs = []
        for f in chunk_files:
            spectrogram = np.load(f)
            
            spectrogram = self.apply_augmentation(spectrogram, label_onehot, p=0.5) # Apply augmentation to chunk here
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0) # shape: [1, 128, time]
            spectrogram = self.apply_masking(spectrogram, mask_percent=0.1, p=0.5)
            spectrogram = self.normalize(spectrogram) # normalize the spectrograms for training
            inputs.append(spectrogram) 

        return inputs, label_idx, label_onehot  # inputs is a list of tensors

    def apply_augmentation(self, spectrogram, label_onehot, p):
        if self.augment:
            spectrogram, label_onehot = self.mixup(spectrogram, label_onehot, alpha=1.0, p=p)
            spectrogram = self.inject_background_noise(spectrogram, blend_factor=0.5, p=p)
            spectrogram = self.inject_gaussian_noise(spectrogram, mean=0, min_std=0.05, max_std=0.05, p=p)
            spectrogram = self.vertical_roll(spectrogram, vertical=0.05, p=p)
            spectrogram = self.apply_gain(spectrogram, min_gain_db=-12.0, max_gain_db=12.0, p=p)
        return spectrogram

    def mixup(self, spectrogram, label_onehot, alpha=1.0, p=0.5):
        if random.random() < p:
            mix_idx = np.random.randint(0,len(self)) #get random spectrogram from batch
            mix_path = self.mix_paths[mix_idx]
            mix_spec = np.load(mix_path).astype(np.float32)  # shape: [128, time]
            #mix_spec = self.clip(mix_spec) # clip since this is original spectrogram

            #make one hot label for the mixup spectrogram
            mix_label_idx = self.label_to_index[self.labels[mix_idx]]
            mix_label_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
            mix_label_onehot[mix_label_idx] = 1.0

            lam = np.random.beta(alpha, alpha) #mixup ratio
            spectrogram = lam*spectrogram + (1-lam)*mix_spec
            new_label = lam*label_onehot + (1-lam)*mix_label_onehot

            return spectrogram, new_label
        return spectrogram, label_onehot


    def inject_background_noise(self, spectrogram, blend_factor, p=0.5):
        if random.random() < p:
            noise = np.load(random.choice(self.noise_paths))  # shape: should match spec
            if noise.shape != spectrogram.shape:
                noise = self._resize_noise(noise, spectrogram.shape)
            #add the noise and return
            return spectrogram + (blend_factor * noise)
        return spectrogram

    def _resize_noise(self, noise, target_shape):
        """Resize noise to match the target spectrogram shape."""
        n_mels, time = target_shape
        noise_mels, noise_time = noise.shape

        # Time dimension
        if noise_time < time:
            repeats = (time // noise_time) + 1
            noise = np.tile(noise, (1, repeats))[:, :time]
        elif noise_time > time:
            noise = noise[:, :time]

        # Mel dimension
        if noise_mels < n_mels:
            noise = np.pad(noise, ((0, n_mels - noise_mels), (0, 0)))
        elif noise_mels > n_mels:
            noise = noise[:n_mels, :]

        return noise

    def inject_gaussian_noise(self, spectrogram, mean=0, min_std=0.001, max_std=0.015, p=0.5):
        if random.random() < p:
            std = np.random.uniform(min_std, max_std)
            noise = np.random.normal(mean, std, size=spectrogram.shape).astype(np.float32)
            return spectrogram + noise
        return spectrogram
    
    def vertical_roll(self, spectrogram, vertical=0.1, p=0.5):
        if random.random() < p:
            v_shift = int(spectrogram.shape[0] * random.uniform(-vertical, vertical))
            spectrogram = np.roll(spectrogram, v_shift, axis=0)
            return spectrogram
        return spectrogram


    def apply_gain(self, spectrogram, min_gain_db=-12.0, max_gain_db=12.0, p=0.5):
        if random.random() < p:
            gain_db = random.uniform(min_gain_db, max_gain_db)
            gain_linear = 10 ** (gain_db / 20)
            return spectrogram * gain_linear
        return spectrogram

    def apply_masking(self, spectrogram, mask_percent=0.1, p=0.5):
        """Takes spectrogram as tensor. Applies masking from torchaudio library"""
        width = spectrogram.shape[-1]
        if random.random() < p:
            time_mask_param = mask_percent * width
            freq_mask_param = mask_percent * 128

            time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
            freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

            spectrogram = freq_mask(spectrogram)
            spectrogram = time_mask(spectrogram)

        return spectrogram
            
    def normalize(self, spectrogram):
        #normalize data
        mean = spectrogram.mean()
        std = spectrogram.std()
        if std > 0:
            spectrogram = (spectrogram - mean) / std
        else:
            spectrogram = spectrogram - mean  # Avoid divide-by-zero
        
        return spectrogram




def collate_fn(batch):
    """
    !!! Must be used with batch_size=1 
    Collation function for chunked spectrograms.
    Stacks all the chunks belonging to a single file into tensot for the batch
    Returns single label for the whole bath
    """
    batch_inputs, batch_labels, batch_onehot_label = zip(*batch)

    # Concatenate all chunks from all batch items
    # inputs: list of list of tensors → flatten into single list
    all_chunks = [chunk for chunks in batch_inputs for chunk in chunks]
    inputs = torch.stack(all_chunks)  # shape: (total_chunks, ...)
    #inputs = inputs.unsqueeze(1)
    # Keep original labels (no repeating)
    labels = torch.tensor(batch_labels, dtype=torch.long)
    onehot_label = torch.stack(batch_onehot_label)
    

    return inputs, labels, onehot_label



#get data loaders
def get_dataloader(dataframe, base_path, noise_paths, batch_size, label_to_index, max_frames, augment):
    
    dataset = BirdSpectrogramDataset(dataframe, 
                                    base_path=base_path,
                                    noise_paths=noise_paths, 
                                    label_to_index=label_to_index,
                                    max_frames=max_frames,
                                    augment=augment,
                                    )

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True, 
                            drop_last=False)

    return dataloader

def get_chunk_dataloader(dataframe, base_path, noise_paths, mix_paths, label_to_index, batch_size=1, augment=False):

    dataset = ChunkSpectrogramDataset(dataframe, base_path, noise_paths, mix_paths, label_to_index, augment)
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=True, 
                            num_workers=4, 
                            drop_last=False, 
                            collate_fn=collate_fn)

    return dataloader