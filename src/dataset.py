import os
import numpy as np
import torch
from torch.utils.data import Dataset

class DriverStateDataset(Dataset):
    """
    PyTorch Dataset for loading sequence data for CNN-LSTM.
    Loads pre-processed numpy arrays containing:
        - eye_left: shape (N, seq_len, 64, 64, 3)
        - eye_right: shape (N, seq_len, 64, 64, 3)
        - mouth: shape (N, seq_len, 64, 64, 3)
        - geom: shape (N, seq_len, 10)
        - labels: shape (N, 1) or (N,)
    """
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (str): Path to the saved numpy file (.npz)
            transform (callable, optional): Optional transform to be applied on image sequences
        """
        self.transform = transform
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}. Run preprocessing first or generate dummy data.")
            
        data = np.load(data_path)
        
        # Load and convert to standard shapes
        # Images are saved as uint8 (0-255) to save space, we normalize to float32 (0-1) later.
        self.eye_left = data['eye_left']
        self.eye_right = data['eye_right']
        self.mouth = data['mouth']
        self.geom = data['geom'].astype(np.float32)
        self.labels = data['labels'].astype(np.float32)
        
        # Ensure labels have shape (N, 1)
        if len(self.labels.shape) == 1:
            self.labels = np.expand_dims(self.labels, axis=-1)
            
        self.num_samples = len(self.labels)
        print(f"Loaded dataset from {data_path}:")
        print(f"  - Samples: {self.num_samples}")
        print(f"  - Eye Left shape: {self.eye_left.shape}")
        print(f"  - Eye Right shape: {self.eye_right.shape}")
        print(f"  - Mouth shape: {self.mouth.shape}")
        print(f"  - Geometric shape: {self.geom.shape}")
        print(f"  - Labels shape: {self.labels.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Extract images at index
        # Shape: (seq_len, H, W, C)
        el_img = self.eye_left[idx]
        er_img = self.eye_right[idx]
        m_img = self.mouth[idx]
        geom = self.geom[idx]
        label = self.labels[idx]
        
        # Convert images to (seq_len, C, H, W) and normalize to [0, 1]
        # Transpose from (seq_len, H, W, C) to (seq_len, C, H, W)
        el_tensor = torch.from_numpy(el_img).permute(0, 3, 1, 2).float() / 255.0
        er_tensor = torch.from_numpy(er_img).permute(0, 3, 1, 2).float() / 255.0
        m_tensor = torch.from_numpy(m_img).permute(0, 3, 1, 2).float() / 255.0
        
        # Optional: apply visual transforms per frame if provided
        if self.transform:
            # Note: transform should process a sequence of frames consistently
            # Apply to each frame
            seq_len = el_tensor.size(0)
            el_list = [self.transform(el_tensor[t]) for t in range(seq_len)]
            er_list = [self.transform(er_tensor[t]) for t in range(seq_len)]
            m_list = [self.transform(m_tensor[t]) for t in range(seq_len)]
            
            el_tensor = torch.stack(el_list)
            er_tensor = torch.stack(er_list)
            m_tensor = torch.stack(m_list)
            
        geom_tensor = torch.from_numpy(geom)
        label_tensor = torch.from_numpy(label)
        
        return el_tensor, er_tensor, m_tensor, geom_tensor, label_tensor
