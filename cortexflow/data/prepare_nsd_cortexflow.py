
import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib
import argparse

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def scale_within_session(betas):
    """
    Standardize data (Z-score) within session.
    Vital for Normalizing Flow models to have input ~ N(0,1).
    """
    # Original SynBrain scaling: betas = betas / 300 
    # But for Flow, we prefer Z-score.
    
    print('Z-scoring beta weights within this session...')
    mb = np.mean(betas, axis=0, keepdims=True)
    sb = np.std(betas, axis=0, keepdims=True)
    
    # Avoid division by zero
    betas = np.nan_to_num((betas - mb) / np.clip(sb, 1e-8, None))
    
    print(f"  Min: {np.min(betas):.3f}, Max: {np.max(betas):.3f}, Mean: {np.mean(betas):.3f}, Std: {np.mean(np.std(betas, axis=0)):.3f}")
    return betas

def main():
    parser = argparse.ArgumentParser(description='Prepare NSD fMRI data for CortexFlow')
    parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
    parser.add_argument("-session", "--session", help="Number of sessions", default=40, type=int)
    parser.add_argument("--data_root", help="Root directory containing NSD data (nsddata, nsddata_betas)", default="data/NSD")
    parser.add_argument("--out_root", help="Output directory", default="data/processed")
    
    args = parser.parse_args()
    sub = args.sub
    session = args.session
    data_root = args.data_root
    
    assert sub in [1,2,5,7], "Subject must be 1, 2, 5, or 7"
    
    print(f"Processing Subject {sub}, Sessions 1-{session}")
    print(f"Data Root: {data_root}")
    
    # Setup paths
    # Note: Structure expected matches SynBrain's expectation or raw NSD structure
    # Raw NSD usually: nsddata/ppdata/subj01/...
    # SynBrain script used: nsddata/ppdata/subj01/func1pt8mm/roi/
    
    # Helper to handle potential 'subj01' vs 'subj1' naming differences if needed, 
    # but standard NSD is 'subj01'.
    subj_str = f"subj{sub:02d}" 
    
    roi_dir = os.path.join(data_root, 'nsddata', 'ppdata', subj_str, 'func1pt8mm', 'roi')
    betas_dir = os.path.join(data_root, 'nsddata_betas', 'ppdata', subj_str, 'func1pt8mm', 'betas_fithrf_GLMdenoise_RR')
    
    # Load stimulus order design
    # Actual path found: nsddata/experiments/nsd/nsd_expdesign.mat
    stim_order_f = os.path.join(data_root, 'nsddata', 'experiments', 'nsd', 'nsd_expdesign.mat')
    if not os.path.exists(stim_order_f):
        print(f"Error: {stim_order_f} not found.")
        return

    stim_order = loadmat(stim_order_f)
    
    ## Selecting ids for training and test data
    sig_train = {}
    sig_test = {}
    num_trials = session*750
    
    print("Parsing experiment design...")
    for idx in range(num_trials):
        ''' nsdId as in design csv files'''
        # masterordering is 1-based, python is 0-based
        # subjectim is 1-based index into 73k images
        master_idx = stim_order['masterordering'][idx] - 1
        nsdId = stim_order['subjectim'][sub-1, master_idx] - 1
        
        # Split logic from NSD paper/SynBrain: shared 1000 images (nsdId < 1000?? No, nsdId is 0..72999)
        # Actually, shared images are special indices. 
        # But SynBrain simplifies: "if stim_order['masterordering'][idx] > 1000" -> Train?
        # Let's check SynBrain logic again: 
        #   if stim_order['masterordering'][idx]>1000: train
        #   else: test
        # This assumes masterordering 1-1000 are the shared trial instances? 
        # Actually in NSD, masterordering is just the chronological trial number across all subjects?
        # NO. masterordering vector is just 1..30000. 
        # Wait, proper split usually relies on whether the image is shared or unique.
        
        # Let's stick strictly to SynBrain's logic to maintain compatibility for now.
        if stim_order['masterordering'][idx] > 1000:
            if nsdId not in sig_train:
                sig_train[nsdId] = []
            sig_train[nsdId].append(idx)
        else:
            if nsdId not in sig_test:
                sig_test[nsdId] = []
            sig_test[nsdId].append(idx)

    train_im_idx = list(sig_train.keys())
    test_im_idx = list(sig_test.keys())
    
    print(f"Found {len(train_im_idx)} training images and {len(test_im_idx)} test images.")
    
    # Load ROI Mask
    mask_filename = 'nsdgeneral.nii.gz'
    mask_path = os.path.join(roi_dir, mask_filename)
    if not os.path.exists(mask_path):
        print(f"Error: Mask {mask_path} not found.")
        return
        
    print(f"Loading mask from {mask_path}...")
    mask = nib.load(mask_path).get_fdata()
    num_voxel = mask[mask>0].shape[0]
    print(f"Number of voxels in nsdgeneral ROI: {num_voxel}")
    
    # Load and process fMRI data
    fmri = np.zeros((num_trials, num_voxel)).astype(np.float32)
    
    print("Loading beta sessions...")
    for i in range(session):
        beta_filename = f"betas_session{i+1:02d}.nii.gz"
        beta_path = os.path.join(betas_dir, beta_filename)
        
        if not os.path.exists(beta_path):
            print(f"Warning: {beta_path} does not exist. Stopping at session {i}.")
            break
            
        print(f"  Processing {beta_filename}...")
        beta_f = nib.load(beta_path).get_fdata().astype(np.float32)
        
        # Masking and flattening
        betas = beta_f[mask>0].transpose() # [750, num_voxels]
        
        # Scale/Z-score
        fmri[i*750:(i+1)*750] = scale_within_session(betas)
        
        del beta_f
        del betas

    print(f"All fMRI data loaded. Shape: {fmri.shape}")
    
    # Averaging Trials for CortexFlow (Flat Flow input)
    # Target shape: (N, num_voxel) instead of (N, 3, num_voxel)
    # This removes zero-padding artifacts and improves SNR.
    
    print("Organizing and Averaging Training Data...")
    valid_train_indices = []
    fmri_train_list = []
    
    for idx in train_im_idx:
        trials = sorted(sig_train[idx])
        if len(trials) > 0:
            # Average available trials
            mean_response = np.mean(fmri[trials], axis=0)
            fmri_train_list.append(mean_response)
            valid_train_indices.append(idx)
    
    fmri_train = np.array(fmri_train_list, dtype=np.float32)
    print(f"  Final Train Shape: {fmri_train.shape}")

    # Organizing Test Data
    print("Organizing and Averaging Test Data...")
    valid_test_indices = []
    fmri_test_list = []
    
    for idx in test_im_idx:
        trials = sorted(sig_test[idx])
        if len(trials) > 0:
            mean_response = np.mean(fmri[trials], axis=0)
            fmri_test_list.append(mean_response)
            valid_test_indices.append(idx)
            
    fmri_test = np.array(fmri_test_list, dtype=np.float32)
    print(f"  Final Test Shape: {fmri_test.shape}")

    # Save outputs
    out_dir = os.path.join(args.out_root, subj_str)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving to {out_dir}...")
    np.save(os.path.join(out_dir, 'train_fmri_avg.npy'), fmri_train)
    np.save(os.path.join(out_dir, 'test_fmri_avg.npy'), fmri_test)
    
    # Save Image Indices (Updated to only include valid ones)
    np.save(os.path.join(out_dir, 'train_imgIds.npy'), np.array(valid_train_indices))
    np.save(os.path.join(out_dir, 'test_imgIds.npy'), np.array(valid_test_indices))
    
    print("Done!")

if __name__ == "__main__":
    main()
