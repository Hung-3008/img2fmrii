
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CortexDataset(Dataset):
    def __init__(self, data_root, subject="subj01", mode="train"):
        """
        Args:
            data_root (str): Path to processed data (e.g., 'data/processed')
            subject (str): Subject folder name (e.g., 'subj01')
            mode (str): 'train' or 'test'
        """
        self.data_dir = os.path.join(data_root, subject)
        self.mode = mode
        
        # Determine file names
        if mode == "train":
            fmri_file = "train_fmri_avg.npy"
            # Prefer multilayer if exists
            if os.path.exists(os.path.join(self.data_dir, "train_clip_multilayer.npy")):
                clip_file = "train_clip_multilayer.npy"
            else:
                clip_file = "train_clip_vitl14.npy"
        else:
            fmri_file = "test_fmri_avg.npy"
            if os.path.exists(os.path.join(self.data_dir, "test_clip_multilayer.npy")):
                clip_file = "test_clip_multilayer.npy"
            else:
                clip_file = "test_clip_vitl14.npy"
            
        self.fmri_path = os.path.join(self.data_dir, fmri_file)
        self.clip_path = os.path.join(self.data_dir, clip_file)
        
        # Load Data
        print(f"[{mode.upper()}] Loading data from {self.data_dir}...")
        try:
            get_fmri = np.load(self.fmri_path).astype(np.float32)
            get_clip = np.load(self.clip_path).astype(np.float32)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {e}. Did you run prepare_nsd_cortexflow.py?")

        # Check alignment
        assert get_fmri.shape[0] == get_clip.shape[0], \
            f"Mismatch: fMRI {get_fmri.shape[0]} vs CLIP {get_clip.shape[0]}"
            
        self.fmri_data = torch.from_numpy(get_fmri)
        self.clip_data = torch.from_numpy(get_clip)
        
        print(f"  Loaded {len(self.fmri_data)} samples.")
        print(f"  fMRI Dim: {self.fmri_data.shape[1]}, CLIP Dim: {self.clip_data.shape[1]}")

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        # Returns: fmri_voxel_vector, clip_embedding
        return self.fmri_data[idx], self.clip_data[idx]

if __name__ == "__main__":
    # Quick Test
    import sys
    root = "data/processed"
    if len(sys.argv) > 1:
        root = sys.argv[1]
        
    try:
        ds = CortexDataset(root, mode="train")
        f, c = ds[0]
        print(f"Sample 0: fMRI={f.shape}, CLIP={c.shape}")
    except Exception as e:
        print(f"Test failed: {e}")
