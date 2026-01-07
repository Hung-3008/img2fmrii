"""
CortexFlow Final Evaluation

Combines:
1. SynBrain compatibility (3-trial comparison) from evaluate.py
2. Advanced inference (Multi-sample, Hybrid, Residual) from evaluate_v2.py
"""

import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from dataset import CortexDataset
from autoencoder import BrainAutoencoder, BrainVAE
from model import CortexFlow
from hybrid_sampler import HybridPredictor


def get_args():
    parser = argparse.ArgumentParser(description="Evaluating CortexFlow (Final)")
    parser.add_argument("--data_root", default="data/NSD/nsd")
    parser.add_argument("--subject", default="subj01")
    
    # Checkpoints
    parser.add_argument("--flow_path", required=True)
    parser.add_argument("--reg_path", required=True)
    parser.add_argument("--ae_path", required=True)
    parser.add_argument("--ae_type", default="vae", choices=["ae", "vae"])
    
    # Model Config
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--clip_dim", type=int, default=3840) # Vit-L/14 + Multi-layer
    
    # Inference Config
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--residual_flow", action="store_true")
    
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default=None)
    
    return parser.parse_args()


def compute_synbrain_metrics(pred, target_3trial):
    """
    pred: [B, D]
    target_3trial: [B, 3, D]
    """
    B = pred.shape[0]
    mse_vals = []
    pearson_vals = []
    cos_vals = []
    
    pred_np = pred.cpu().numpy()
    target_np = target_3trial.cpu().numpy()
    
    for i in range(B):
        p = pred_np[i]
        for j in range(3):
            t = target_np[i, j]
            
            # MSE
            mse = np.mean((p - t) ** 2)
            mse_vals.append(mse)
            
            # Pearson
            p_std = np.std(p)
            t_std = np.std(t)
            if p_std < 1e-6 or t_std < 1e-6:
                # Debug only first few times
                if len(pearson_vals) < 5: 
                    print(f"    [Warning] Constant input at idx {i}, trial {j}. Pred Std: {p_std:.6f}, GT Std: {t_std:.6f}")
            else:
                try:
                    corr, _ = pearsonr(p, t)
                    if not np.isnan(corr):
                        pearson_vals.append(corr)
                except:
                    pass
            
            # Cosine
            cos = cosine_similarity(p.reshape(1, -1), t.reshape(1, -1))[0, 0]
            cos_vals.append(cos)
                
    return {
        'mse': np.mean(mse_vals),
        'pearson': np.mean(pearson_vals) if pearson_vals else 0.0,
        'pearson_std': np.std(pearson_vals) if pearson_vals else 0.0,
        'cosine': np.mean(cos_vals)
    }


def evaluate(args):
    device = torch.device(args.device)
    print(f"Running Final Evaluation on {device}")
    
    # 1. Load Test Data (CLIP)
    # We use CortexDataset to get CLIP, but we'll load 3-trial fMRI manually
    ds = CortexDataset(args.data_root, args.subject, mode="test")
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Load 3-trial Ground Truth
    # Path logic: data/NSD/nsd/subj01/nsd_test_fmri_scale_sub1.npy
    sub_num = args.subject.replace('subj', '').lstrip('0')
    gt_path = os.path.join(args.data_root, args.subject, f"nsd_test_fmri_scale_sub{sub_num}.npy")
    
    if not os.path.exists(gt_path):
        print(f"Error: 3-trial data not found at {gt_path}")
        return
        
    print(f"Loading 3-trial GT from {gt_path}...")
    fmri_gt = np.load(gt_path).astype(np.float32) # [N, 3, D] or [N*3, D]
    
    if len(fmri_gt.shape) == 2:
        N3, D = fmri_gt.shape
        fmri_gt = fmri_gt.reshape(-1, 3, D)
    
    fmri_gt = torch.from_numpy(fmri_gt).to(device)
    print(f"GT Shape: {fmri_gt.shape}") # [N, 3, D]
    
    # 3. Load Models
    # AE
    if args.ae_type == "vae":
        # Pass hidden_dims=None explicitly or use kwargs for latent_dim
        ae = BrainVAE(
            input_dim=15724, 
            clip_dim=768, 
            latent_dim=args.latent_dim
        ).to(device)
    else:
        ae = BrainAutoencoder(
            input_dim=15724, 
            latent_dim=args.latent_dim
        ).to(device)
    
    # Check dims from data
    # We need to actully init AE with correct dims from dataset
    ae_input_dim = ds.fmri_data.shape[1]
    ae_clip_dim = ds.clip_data.shape[1]
    
    if args.ae_type == "vae":
        ae = BrainVAE(
            input_dim=ae_input_dim, 
            clip_dim=ae_clip_dim, 
            latent_dim=args.latent_dim
        ).to(device)
    else:
        ae = BrainAutoencoder(
            input_dim=ae_input_dim, 
            latent_dim=args.latent_dim
        ).to(device)
        
    ae.load_state_dict(torch.load(args.ae_path, map_location=device))
    ae.eval()
    
    # Flow & Regressor
    flow = CortexFlow(args.latent_dim, ae_clip_dim, args.num_layers, args.hidden_dim).to(device)
    flow.load_state_dict(torch.load(args.flow_path, map_location=device))
    flow.eval()
    
    import torch.nn as nn
    regressor = nn.Sequential(
        nn.Linear(ae_clip_dim, 2048),
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
    
    # Hybrid Helper
    hybrid = HybridPredictor(regressor, flow, ae.decoder, args.alpha, args.temperature)
    
    # 4. Inference Loop
    modes = ['mean_only', 'flow_only', 'hybrid', 'multi_sample']
    if args.residual_flow:
        modes.append('residual')
        
    results = {}
    
    for mode in modes:
        print(f"\nScanning {mode}...")
        all_preds = []
        
        with torch.no_grad():
            for _, clip in tqdm(loader):
                clip = clip.to(device)
                
                # Predict
                if mode == 'residual' and args.residual_flow:
                    z_mean = regressor(clip)
                    noise = torch.randn_like(z_mean) * args.temperature
                    z_res = flow.inverse(noise, clip)
                    x_pred = ae.decoder(z_mean + z_res)
                else:
                    target_mode = mode if mode != 'residual' else 'hybrid'
                    x_pred = hybrid.predict_fmri(clip, mode=target_mode, num_samples=args.num_samples)
                
                all_preds.append(x_pred)
                
        all_preds = torch.cat(all_preds, dim=0)
        
        # Truncate GT if mismatch (e.g. 982 test images vs 1000 in GT)
        # Typically ds has fewer samples if some were invalid
        n_pred = all_preds.shape[0]
        n_gt = fmri_gt.shape[0]
        
        if n_pred != n_gt:
            n_min = min(n_pred, n_gt)
            # We assume aligned from start. 
            # Note: prepare_script filtered ids, so we should be careful. 
            # For this script we assume 1-1 mapping validation was done or we just truncate.
            metrics = compute_synbrain_metrics(all_preds[:n_min], fmri_gt[:n_min])
        else:
            metrics = compute_synbrain_metrics(all_preds, fmri_gt)
            
        results[mode] = metrics
        print(f"  MSE: {metrics['mse']:.6f} | Pearson: {metrics['pearson']:.4f} | Cosine: {metrics['cosine']:.4f}")

    # Save
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "final_results.txt"), "w") as f:
            f.write(f"Evaluation (Temp={args.temperature}, Samples={args.num_samples}, Alpha={args.alpha})\n\n")
            for m, res in results.items():
                f.write(f"{m:<15} MSE: {res['mse']:.6f}  Pearson: {res['pearson']:.4f}  Cosine: {res['cosine']:.4f}\n")
        print(f"\nSaved to {args.save_dir}/final_results.txt")


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
