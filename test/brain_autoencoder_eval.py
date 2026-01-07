import os
import sys
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# Add cortexflow/src to PYTHONPATH so we can reuse dataset & model
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORTEXFLOW_SRC = os.path.join(PROJECT_ROOT, "cortexflow", "src")
if CORTEXFLOW_SRC not in sys.path:
    sys.path.append(CORTEXFLOW_SRC)

from dataset import CortexDataset  # noqa: E402
from autoencoder import BrainAutoencoder  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BrainAutoencoder vs zero baseline")
    parser.add_argument("--data_root", type=str, default="data/processed",
                        help="Root folder of processed fMRI data")
    parser.add_argument("--subject", type=str, default="subj01",
                        help="Subject ID, e.g. subj01")
    parser.add_argument("--ckpt", type=str, default="checkpoints/brain_ae/best_ae.pth",
                        help="Path to trained BrainAutoencoder checkpoint")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for evaluation")
    parser.add_argument("--latent_dim", type=int, default=1024,
                        help="Latent dimension (must match training)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout used in BrainAutoencoder (must match training)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force evaluation on CPU even if CUDA is available")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Load test dataset
    test_ds = CortexDataset(args.data_root, args.subject, mode="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Basic data stats
    fmri_all = test_ds.fmri_data  # [N, D], on CPU
    data_mean = fmri_all.mean().item()
    data_std = fmri_all.std().item()
    data_min = fmri_all.min().item()
    data_max = fmri_all.max().item()
    print(f"Data Stats (test) - Mean: {data_mean:.4f}, Std: {data_std:.4f}, "
          f"Min: {data_min:.4f}, Max: {data_max:.4f}")

    # Zero baseline: always predict 0
    zero_mse = (fmri_all ** 2).mean().item()
    print(f"Zero baseline MSE (predict 0): {zero_mse:.4f}")

    # Prepare model
    fmri_dim = fmri_all.shape[1]
    model = BrainAutoencoder(
        input_dim=fmri_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)

    if not os.path.exists(args.ckpt):
        print(f"ERROR: checkpoint not found at {args.ckpt}")
        return

    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded BrainAutoencoder from {args.ckpt}")

    total_mse = 0.0
    total_pearson = 0.0
    n_batches = 0

    # Track reconstruction stats from first batch for sanity check
    first_batch_recon_stats_printed = False

    with torch.no_grad():
        for fmri, _ in test_loader:
            fmri = fmri.to(device)

            recon, z = model(fmri)

            # MSE per batch
            mse = F.mse_loss(recon, fmri)
            total_mse += mse.item()

            # Pearson per batch (same style as train_ae.py)
            vx = fmri - fmri.mean(dim=1, keepdim=True)
            vy = recon - recon.mean(dim=1, keepdim=True)
            costheta = (vx * vy).sum(dim=1) / (
                torch.sqrt((vx ** 2).sum(dim=1)) *
                torch.sqrt((vy ** 2).sum(dim=1)) + 1e-8
            )
            total_pearson += costheta.mean().item()

            if not first_batch_recon_stats_printed:
                r_mean = recon.mean().item()
                r_std = recon.std().item()
                r_min = recon.min().item()
                r_max = recon.max().item()
                print(f"Recon Stats (first batch) - Mean: {r_mean:.4f}, Std: {r_std:.4f}, "
                      f"Min: {r_min:.4f}, Max: {r_max:.4f}")
                first_batch_recon_stats_printed = True

            n_batches += 1

    if n_batches == 0:
        print("No batches in DataLoader; nothing to evaluate.")
        return

    avg_mse = total_mse / n_batches
    avg_pearson = total_pearson / n_batches

    print("\n=== BrainAutoencoder Evaluation (test split) ===")
    print(f"Zero Baseline MSE:   {zero_mse:.4f}")
    print(f"AE Test MSE:         {avg_mse:.4f}")
    print(f"AE Test Pearson:     {avg_pearson:.4f}")
    print("==============================================")


if __name__ == "__main__":
    main()
