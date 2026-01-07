import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import CortexDataset
from autoencoder import BrainVAE
from model import CortexFlow
from hybrid_sampler import HybridPredictor

def visualize(args):
    device = torch.device(args.device)
    print(f"Running Visualization on {device}")

    # 1. Load Data
    print("Loading Data...")
    
    subj_dir = os.path.join(args.data_root, args.subject)
    
    # Load 3-trial Raw fMRI
    sub_num = args.subject.replace('subj', '').lstrip('0') or '1'
    raw_fmri_path = os.path.join(subj_dir, f"nsd_test_fmri_scale_sub{sub_num}.npy")
    
    print(f"Loading raw 3-trial fMRI form {raw_fmri_path}...")
    try:
        raw_fmri_3trial = np.load(raw_fmri_path) # [N, 3, V]
        print(f"Raw fMRI Shape: {raw_fmri_3trial.shape}")
    except FileNotFoundError:
        print("Error: 3-trial fMRI file not found.")
        return

    # Load Stimuli Images
    stim_path = os.path.join(subj_dir, f"nsd_test_stim_sub{sub_num}.npy")
    print(f"Loading stimuli from {stim_path}...")
    try:
        stim_images = np.load(stim_path) # [N, 425, 425, 3] or similar
        print(f"Stimuli Shape: {stim_images.shape}")
    except FileNotFoundError:
        print(f"Error: Stimuli file {stim_path} not found.")
        return
        
    # Load CLIP embeddings (via Dataset or manual)
    # Using manual loading to be sure we match indices
    clip_path = os.path.join(subj_dir, f"nsd_test_clip_sub{sub_num}.npy")
    print(f"Loading CLIP from {clip_path}...")
    try:
        clip_data = np.load(clip_path) # [N, D]
        print(f"CLIP Shape: {clip_data.shape}")
    except FileNotFoundError:
        print(f"Error: CLIP file {clip_path} not found.")
        return

    # 2. Load Models
    print("Loading Models...")
    input_dim = raw_fmri_3trial.shape[2]
    clip_dim = clip_data.shape[1]
    
    # VAE
    ae = BrainVAE(input_dim=input_dim, clip_dim=clip_dim, latent_dim=args.latent_dim).to(device)
    ae.load_state_dict(torch.load(args.ae_path, map_location=device))
    ae.eval()
    
    # Flow
    flow = CortexFlow(args.latent_dim, clip_dim, args.num_layers, args.hidden_dim).to(device)
    flow.load_state_dict(torch.load(args.flow_path, map_location=device))
    flow.eval()
    
    # Regressor
    import torch.nn as nn
    regressor = nn.Sequential(
        nn.Linear(clip_dim, 2048),
        nn.LayerNorm(2048),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(2048, 1024),
        nn.LayerNorm(1024),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(1024, args.latent_dim)
    ).to(device)
    regressor.load_state_dict(torch.load(args.reg_path, map_location=device))
    regressor.eval()
    
    # Hybrid Predictor
    hybrid = HybridPredictor(regressor, flow, ae.decoder, alpha=0.3, temperature=0.8)
    
    # 3. Select 3 Samples
    num_samples = len(raw_fmri_3trial)
    # Ensure alignment
    min_len = min(len(stim_images), len(clip_data), num_samples)
    
    indices = np.random.choice(min_len, 3, replace=False)
    indices = sorted(indices)
    
    # 4. Synthesize & Plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, idx in enumerate(indices):
        print(f"Processing Sample {i+1}/3 (Index {idx})...")
        
        # Get Data
        clip_emb = torch.from_numpy(clip_data[idx]).unsqueeze(0).to(device).float()
        
        # Get Image
        img = stim_images[idx]
        
        # Get Raw fMRI (3 trials)
        raw_traces = raw_fmri_3trial[idx] # [3, V]
        
        # Synthesize
        with torch.no_grad():
            recon = hybrid.predict_fmri(clip_emb, mode='hybrid', num_samples=1)
            recon = recon.cpu().numpy().squeeze() # [V]
            
        # Plotting
        ax_img = axes[i, 0]
        ax_raw = axes[i, 1]
        ax_syn = axes[i, 2]
        
        # Image
        ax_img.imshow(img.astype(np.uint8))
        ax_img.set_title(f"Visual Stimulus (Idx {idx})")
        ax_img.axis('off')
        
        # Raw fMRI
        colors = ['#1f77b4', '#9467bd', '#2ca02c'] # Blue, Purple, Green-ish
        for trial_i in range(3):
            ax_raw.plot(raw_traces[trial_i], color=colors[trial_i], alpha=0.6, linewidth=0.5)
        
        ax_raw.set_title("Raw fMRI (3-Trial)")
        ax_raw.set_xlim(0, len(recon))
        ax_raw.axis('off')

        # Synthesized
        ax_syn.plot(recon, color='orange', linewidth=0.8)
        ax_syn.set_title("Synthesized fMRI")
        ax_syn.set_xlim(0, len(recon))
        ax_syn.set_ylim(ax_raw.get_ylim()) # Match scale
        
        ax_syn.axis('off')

    plt.tight_layout()
    save_path = "visualize_synthesis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Paths (Defaults based on common structure)
    parser.add_argument("--data_root", default="../../data/NSD/nsd")
    parser.add_argument("--subject", default="subj01")
    
    # Checkpoints
    parser.add_argument("--ae_path", default="../../checkpoints/cortexflow_vae/best_ae.pth")
    parser.add_argument("--flow_path", default="../../checkpoints/cortexflow_v/best_flow.pth")
    parser.add_argument("--reg_path", default="../../checkpoints/cortexflow_v/best_regressor.pth")
    
    # Config
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    visualize(args)
