import numpy as np
import os
import sys

# Define paths
data_dir = "/media/hung/data1/codes/synfmri/data/processed/subj01"
norm_path = "/media/hung/data1/codes/synfmri/SynBrain/src/mindeye2/norm_mean_scale_sub1.npz"

# Load files
try:
    print(f"Loading fMRI from {data_dir}/test_fmri_avg.npy")
    test_fmri = np.load(os.path.join(data_dir, "test_fmri_avg.npy"))
    print(f"Shape: {test_fmri.shape}")
    
    # Calculate stats
    fmri_mean = np.mean(test_fmri)
    fmri_std = np.std(test_fmri)
    print(f"Original fMRI: Mean={fmri_mean:.4f}, Std={fmri_std:.4f}, Min={np.min(test_fmri):.4f}, Max={np.max(test_fmri):.4f}")
    
    # Calculate /3.0 stats
    fmri_div3 = test_fmri / 3.0
    div3_mean = np.mean(fmri_div3)
    div3_std = np.std(fmri_div3)
    print(f"Scaled / 3.0:  Mean={div3_mean:.4f}, Std={div3_std:.4f}, Min={np.min(fmri_div3):.4f}, Max={np.max(fmri_div3):.4f}")

except Exception as e:
    print(f"Error loading fMRI: {e}")

try:
    path = os.path.join(data_dir, "nsd_test_fmri_scale_sub1.npy")
    print(f"Loading Raw Scale fMRI from {path}")
    raw_fmri = np.load(path)
    print(f"Shape: {raw_fmri.shape}") # Should be (1000, 3, voxels)
    
    # Check stats of raw
    raw_mean = np.mean(raw_fmri)
    raw_std = np.std(raw_fmri)
    print(f"Raw Scale fMRI: Mean={raw_mean:.4f}, Std={raw_std:.4f}")
    
    # Check stats of mean(raw, axis=1)
    raw_avg = np.mean(raw_fmri, axis=1)
    avg_mean = np.mean(raw_avg)
    avg_std = np.std(raw_avg)
    print(f"Mean(Raw, axis=1): Mean={avg_mean:.4f}, Std={avg_std:.4f}")

except Exception as e:
    print(f"Error loading Raw Scale fMRI: {e}")


try:
    print(f"Loading Norm Params from {norm_path}")
    norm_params = np.load(norm_path)
    norm_mean = norm_params['mean']
    norm_scale = norm_params['scale']
    
    print(f"Norm Mean:     Mean={np.mean(norm_mean):.4f}, Std={np.std(norm_mean):.4f}, Min={np.min(norm_mean):.4f}, Max={np.max(norm_mean):.4f}")
    print(f"Norm Scale:    Mean={np.mean(norm_scale):.4f}, Std={np.std(norm_scale):.4f}, Min={np.min(norm_scale):.4f}, Max={np.max(norm_scale):.4f}")
    
except Exception as e:
    print(f"Error loading Norm Params: {e}")
