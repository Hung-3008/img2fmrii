
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CortexDataset(Dataset):
    def __init__(self, data_root, subject="subj01", mode="train"):
        """
        Args:
            data_root (str): Path to processed data (e.g., 'data/processed' or 'data/NSD/nsd')
            subject (str): Subject folder name (e.g., 'subj01')
            mode (str): 'train' or 'test'
        """
        self.data_dir = os.path.join(data_root, subject)
        self.mode = mode
        
        # Extract subject number for SynBrain format
        sub_num = subject.replace('subj', '').lstrip('0') or '1'
        
        # Try CortexFlow format first, then fall back to SynBrain format
        if mode == "train":
            # CortexFlow format
            fmri_file_cf = "train_fmri_avg.npy"
            clip_file_cf = "train_clip_multilayer.npy"
            clip_file_cf_alt = "train_clip_vitl14.npy"
            # SynBrain format
            fmri_file_sb = f"nsd_train_fmri_scale_sub{sub_num}.npy"
            clip_file_sb = f"nsd_train_clip_sub{sub_num}.npy"
        else:
            # CortexFlow format
            fmri_file_cf = "test_fmri_avg.npy"
            clip_file_cf = "test_clip_multilayer.npy"
            clip_file_cf_alt = "test_clip_vitl14.npy"
            # SynBrain format
            fmri_file_sb = f"nsd_test_fmri_scale_sub{sub_num}.npy"
            clip_file_sb = f"nsd_test_clip_sub{sub_num}.npy"
        
        # Determine which format to use
        fmri_path_cf = os.path.join(self.data_dir, fmri_file_cf)
        fmri_path_sb = os.path.join(self.data_dir, fmri_file_sb)
        
        if os.path.exists(fmri_path_cf):
            # Use CortexFlow format
            self.fmri_path = fmri_path_cf
            if os.path.exists(os.path.join(self.data_dir, clip_file_cf)):
                self.clip_path = os.path.join(self.data_dir, clip_file_cf)
            else:
                self.clip_path = os.path.join(self.data_dir, clip_file_cf_alt)
            self.data_format = "cortexflow"
        elif os.path.exists(fmri_path_sb):
            # Use SynBrain format
            self.fmri_path = fmri_path_sb
            self.clip_path = os.path.join(self.data_dir, clip_file_sb)
            self.data_format = "synbrain"
        else:
            raise FileNotFoundError(
                f"No data found in {self.data_dir}. "
                f"Tried: {fmri_file_cf} (CortexFlow) and {fmri_file_sb} (SynBrain)"
            )
        
        # Load Data
        print(f"[{mode.upper()}] Loading data from {self.data_dir} ({self.data_format} format)...")
        try:
            get_fmri = np.load(self.fmri_path).astype(np.float32)
            get_clip = np.load(self.clip_path).astype(np.float32)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {e}")
        
        # Handle SynBrain 3-trial format: [N, 3, D] -> [N, D] by averaging
        if len(get_fmri.shape) == 3:
            print(f"  Converting 3-trial format {get_fmri.shape} to averaged...")
            get_fmri = np.mean(get_fmri, axis=1)

        # Handle mismatch by truncating to minimum length
        n_fmri = get_fmri.shape[0]
        n_clip = get_clip.shape[0]
        if n_fmri != n_clip:
            min_n = min(n_fmri, n_clip)
            print(f"  Warning: Sample mismatch (fMRI={n_fmri}, CLIP={n_clip}). Truncating to {min_n}.")
            get_fmri = get_fmri[:min_n]
            get_clip = get_clip[:min_n]
            
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
