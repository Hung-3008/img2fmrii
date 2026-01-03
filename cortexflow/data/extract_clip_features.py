
import os
import torch
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
import argparse

try:
    import clip
except ImportError:
    print("Please install clip: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Extract CLIP ViT-L/14 Features for NSD')
    parser.add_argument("--sub", help="Subject Number (to find split)", default=1, type=int)
    parser.add_argument("--data_root", help="Root NSD data", default="data/NSD")
    parser.add_argument("--processed_root", help="Directory where prepare_nsd_cortexflow saved outputs", default="data/processed")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    subj_str = f"subj{args.sub:02d}"
    proc_dir = os.path.join(args.processed_root, subj_str)
    
    # Load Image IDs from preprocessing step
    try:
        train_ids = np.load(os.path.join(proc_dir, 'train_imgIds.npy'))
        test_ids = np.load(os.path.join(proc_dir, 'test_imgIds.npy'))
    except FileNotFoundError:
        print(f"Error: Could not find imgIds.npy in {proc_dir}. Run prepare_nsd_cortexflow.py first.")
        return

    # Load Stimuli
    # Actual path found: nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5
    stim_path = os.path.join(args.data_root, 'nsddata_stimuli', 'stimuli', 'nsd', 'nsd_stimuli.hdf5')
    if not os.path.exists(stim_path):
        print(f"Error: {stim_path} not found.")
        return
        
    print(f"Opening stimuli from {stim_path}...")
    f_stim = h5py.File(stim_path, 'r')
    stimuli = f_stim['imgBrick'] # Lazy load
    
    # Load CLIP model
    print(f"Loading CLIP ViT-L/14 on {args.device}...")
    model, preprocess = clip.load("ViT-L/14", device=args.device)
    model.eval()
    
    def extract_features(img_ids, mode_name):
        print(f"Extracting features for {mode_name} ({len(img_ids)} images)...")
        features_list = []
        
        # Batch processing
        for i in tqdm(range(0, len(img_ids), args.batch_size)):
            batch_ids = img_ids[i : i + args.batch_size]
            batch_images = []
            
            for idx in batch_ids:
                # NSD images are (425, 425, 3), usually standard RGB
                # HDF5 indexing
                img_array = stimuli[idx]
                img = Image.fromarray(img_array.astype('uint8'))
                batch_images.append(preprocess(img))
            
            # Stack and move to device
            batch_input = torch.stack(batch_images).to(args.device)
            
            with torch.no_grad():
                # Encode image
                image_features = model.encode_image(batch_input)
                # Normalize (important for CLIP covariance)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
            features_list.append(image_features.cpu().numpy())
            
        return np.concatenate(features_list, axis=0)

    # Extract Train
    train_feats = extract_features(train_ids, "Train")
    np.save(os.path.join(proc_dir, 'train_clip_vitl14.npy'), train_feats)
    print(f"Saved train features: {train_feats.shape}")
    
    # Extract Test
    test_feats = extract_features(test_ids, "Test")
    np.save(os.path.join(proc_dir, 'test_clip_vitl14.npy'), test_feats)
    print(f"Saved test features: {test_feats.shape}")
    
    print("Done!")

if __name__ == "__main__":
    main()
