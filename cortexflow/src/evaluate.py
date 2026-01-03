
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from dataset import CortexDataset
from model import CortexFlow

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate CortexFlow - SynBrain Compatible")
    parser.add_argument("--data_root", default="data/processed", help="Path to processed data")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument("--model_path", required=True, help="Path to best_model.pth")
    
    # Model Params (Must match training)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=100)
    
    # Evaluation mode
    parser.add_argument("--eval_mode", type=str, default="both", 
                        choices=["reconstruction", "sampling", "both"],
                        help="Evaluation mode: reconstruction, sampling, or both")
    
    # SynBrain compatibility
    parser.add_argument("--use_trials", action="store_true",
                        help="Use 3-trial format [N, 3, D] for SynBrain-compatible evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed debug information")
    
    return parser.parse_args()


def compute_cka(X, Y):
    """
    Compute Centered Kernel Alignment (CKA) between X and Y.
    From SynBrain eval.py
    
    Args:
        X: [N, D] tensor
        Y: [N, D] tensor
    Returns:
        CKA score (float)
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    dot_XY = torch.norm(X @ Y.T) ** 2
    dot_XX = torch.norm(X @ X.T) ** 2
    dot_YY = torch.norm(Y @ Y.T) ** 2
    return (dot_XY / (torch.sqrt(dot_XX * dot_YY) + 1e-8)).item()


def evaluate_synbrain_metrics(all_recon_fmri, all_fmri, verbose=False):
    """
    Evaluate voxel-level and structural-level metrics (SynBrain compatible).
    
    Args:
        all_recon_fmri: Tensor [N, 1, D] - predictions
        all_fmri: Tensor [N, 3, D] - ground-truth with 3 trials
        verbose: Print detailed stats
    
    Returns:
        dict with keys: MSE, Pearson, CKA, Cosine
    """
    N, _, D = all_recon_fmri.shape
    assert all_fmri.shape == (N, 3, D), f"Expected all_fmri shape [{N}, 3, {D}], got {all_fmri.shape}"
    
    all_recon = all_recon_fmri.squeeze(1)  # [N, D]
    
    # Voxel-level metrics: compare each prediction to all 3 trials
    mse_vals = []
    pearson_vals = []
    
    for i in range(N):
        recon = all_recon[i]
        for j in range(3):
            target = all_fmri[i, j]
            
            # MSE
            mse = torch.mean((recon - target) ** 2).item()
            mse_vals.append(mse)
            
            # Pearson correlation
            try:
                p = pearsonr(recon.cpu().numpy(), target.cpu().numpy())[0]
                if not np.isnan(p):
                    pearson_vals.append(p)
            except:
                pass
    
    # Structure-level metrics: compare distributions
    # Repeat predictions to match 3 trials: [N, D] -> [N*3, D]
    recon_flat = all_recon.repeat_interleave(3, dim=0)  # [N*3, D]
    target_flat = all_fmri.view(-1, D)                  # [N*3, D]
    
    # CKA
    cka = compute_cka(recon_flat, target_flat)
    
    # Cosine Similarity
    recon_np = recon_flat.cpu().numpy()
    target_np = target_flat.cpu().numpy()
    cos_vals = []
    for i in range(recon_np.shape[0]):
        cs = cosine_similarity(recon_np[i:i+1], target_np[i:i+1])[0, 0]
        cos_vals.append(cs)
    
    results = {
        "MSE": np.mean(mse_vals),
        "Pearson": np.mean(pearson_vals),
        "CKA": cka,
        "Cosine": np.mean(cos_vals)
    }
    
    if verbose:
        print(f"  MSE: {len(mse_vals)} comparisons (N={N}, 3 trials each)")
        print(f"  Pearson: {len(pearson_vals)} valid correlations")
        print(f"  CKA/Cosine: {recon_flat.shape[0]} flattened samples")
    
    return results


def compute_basic_metrics(pred, target, verbose=False):
    """
    Compute basic metrics for [N, D] format (backward compatibility).
    
    Args:
        pred: [N, D] numpy array
        target: [N, D] numpy array
    Returns:
        dict: MSE, Pearson, Cosine
    """
    mse = np.mean((pred - target) ** 2)
    
    pearson_vals = []
    for i in range(len(pred)):
        try:
            p, _ = pearsonr(pred[i], target[i])
            if not np.isnan(p):
                pearson_vals.append(p)
        except:
            pass
    
    cos_vals = []
    for i in range(len(pred)):
        cs = cosine_similarity(pred[i].reshape(1, -1), target[i].reshape(1, -1))[0, 0]
        cos_vals.append(cs)
    
    results = {
        "MSE": mse,
        "Pearson": np.mean(pearson_vals) if pearson_vals else float('nan'),
        "Cosine": np.mean(cos_vals)
    }
    
    if verbose:
        nan_count = len(pred) - len(pearson_vals)
        if nan_count > 0:
            print(f"  Warning: {nan_count} samples returned NaN for Pearson")
    
    return results


def evaluate(args):
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Evaluation mode: {args.eval_mode}")
    if args.use_trials:
        print("Using 3-trial format [N, 3, D] - SynBrain compatible")
    
    # Load Data
    test_ds = CortexDataset(args.data_root, args.subject, mode="test")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    x_dim = test_ds.fmri_data.shape[1]
    c_dim = test_ds.clip_data.shape[1]
    
    # Prepare ground-truth in appropriate format
    if args.use_trials:
        # Simulate 3-trial format by repeating averaged data
        # In production, load actual 3-trial data here
        test_fmri_avg = test_ds.fmri_data.numpy()  # [N, D]
        test_fmri_3trial = test_fmri_avg[:, None, :].repeat(3, axis=1)  # [N, 3, D]
        test_fmri_3trial = torch.from_numpy(test_fmri_3trial).float()
        print(f"Test fMRI (3-trial): {test_fmri_3trial.shape}")
    
    # Load Model
    model = CortexFlow(x_dim=x_dim, c_dim=c_dim, num_layers=args.num_layers, 
                       hidden_dim=args.hidden_dim, dropout=args.dropout)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded model: {args.model_path}")
    else:
        print(f"ERROR: Model not found at {args.model_path}")
        return
    
    model.to(device)
    model.eval()
    
    # Determine modes to run
    modes_to_run = []
    if args.eval_mode == "both":
        modes_to_run = ["reconstruction", "sampling"]
    else:
        modes_to_run = [args.eval_mode]
    
    all_results = {}
    
    for eval_mode in modes_to_run:
        print(f"\n{'='*60}")
        print(f" {eval_mode.upper()} EVALUATION")
        print('='*60)
        
        all_preds = []
        all_targets = []
        all_latents = []
        
        with torch.no_grad():
            for x_gt, c in tqdm(test_loader, desc=eval_mode):
                x_gt, c = x_gt.to(device), c.to(device)
                
                if eval_mode == "reconstruction":
                    # Encode then decode
                    z_encoded, log_det = model(x_gt, c)
                    all_latents.append(z_encoded.cpu())
                    x_pred = model.inverse(z_encoded, c)
                else:  # sampling
                    # Sample from N(0,1)
                    z_sampled = torch.randn_like(x_gt).to(device)
                    all_latents.append(z_sampled.cpu())
                    x_pred = model.inverse(z_sampled, c)
                
                all_preds.append(x_pred.cpu())
                all_targets.append(x_gt.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)      # [N, D]
        all_targets = torch.cat(all_targets, dim=0)  # [N, D]
        all_latents = torch.cat(all_latents, dim=0)  # [N, D]
        
        if args.verbose:
            print(f"\nData shapes:")
            print(f"  Predictions: {all_preds.shape}")
            print(f"  Targets: {all_targets.shape}")
            print(f"  Latents: {all_latents.shape}")
            
            print(f"\nLatent statistics:")
            print(f"  Mean: {all_latents.mean():.6f}, Std: {all_latents.std():.6f}")
            if eval_mode == "reconstruction":
                print(f"  (Should be ~N(0,1) if model trained properly)")
            
            print(f"\nPrediction statistics:")
            print(f"  Mean: {all_preds.mean():.6f}, Std: {all_preds.std():.6f}")
            print(f"  Min: {all_preds.min():.6f}, Max: {all_preds.max():.6f}")
        
        # Compute metrics
        if args.use_trials:
            # SynBrain-compatible: [N, 1, D] vs [N, 3, D]
            all_preds_3d = all_preds.unsqueeze(1)  # [N, 1, D]
            metrics = evaluate_synbrain_metrics(all_preds_3d, test_fmri_3trial, verbose=args.verbose)
        else:
            # Basic metrics: [N, D] vs [N, D]
            metrics = compute_basic_metrics(all_preds.numpy(), all_targets.numpy(), verbose=args.verbose)
        
        print(f"\nResults:")
        print(f"  MSE:         {metrics['MSE']:.6f}")
        print(f"  Pearson:     {metrics['Pearson']:.6f}")
        if 'CKA' in metrics:
            print(f"  CKA:         {metrics['CKA']:.6f}")
        print(f"  Cosine Sim:  {metrics['Cosine']:.6f}")
        
        all_results[eval_mode] = metrics
    
    # Save results
    res_dir = os.path.dirname(args.model_path)
    
    if args.eval_mode == "both":
        if args.use_trials:
            res_path = os.path.join(res_dir, "eval_results_synbrain.txt")
        else:
            res_path = os.path.join(res_dir, "eval_results_comparison.txt")
        
        with open(res_path, "w") as f:
            f.write("="*60 + "\n")
            f.write("CORTEXFLOW EVALUATION RESULTS\n")
            if args.use_trials:
                f.write("(SynBrain-compatible format [N, 3, D])\n")
            f.write("="*60 + "\n\n")
            
            for mode in modes_to_run:
                results = all_results[mode]
                f.write(f"{mode.upper()} MODE:\n")
                f.write(f"  MSE:         {results['MSE']:.6f}\n")
                f.write(f"  Pearson:     {results['Pearson']:.6f}\n")
                if 'CKA' in results:
                    f.write(f"  CKA:         {results['CKA']:.6f}\n")
                f.write(f"  Cosine Sim:  {results['Cosine']:.6f}\n")
                f.write("\n")
    else:
        mode = args.eval_mode
        if args.use_trials:
            res_path = os.path.join(res_dir, f"eval_results_{mode}_synbrain.txt")
        else:
            res_path = os.path.join(res_dir, f"eval_results_{mode}.txt")
        
        results = all_results[mode]
        with open(res_path, "w") as f:
            f.write(f"{mode.upper()} EVALUATION:\n")
            if args.use_trials:
                f.write("(SynBrain-compatible format [N, 3, D])\n")
            f.write(f"MSE: {results['MSE']}\n")
            f.write(f"Pearson: {results['Pearson']}\n")
            if 'CKA' in results:
                f.write(f"CKA: {results['CKA']}\n")
            f.write(f"Cosine: {results['Cosine']}\n")
    
    print(f"\nResults saved to: {res_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
