
import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

from dataset import CortexDataset
from autoencoder import BrainAutoencoder, BrainVAE


def setup_logger(save_dir: str):
    """Create a logger that logs to both console and a .log file."""
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("BrainAutoencoder")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers when re-running
    if logger.hasHandlers():
        logger.handlers.clear()

    log_path = os.path.join(save_dir, "train_ae.log")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        x = output
        y = target
        vx = x - x.mean(dim=1, keepdim=True)
        vy = y - y.mean(dim=1, keepdim=True)
        costheta = (vx * vy).sum(dim=1) / (torch.sqrt((vx ** 2).sum(dim=1)) * torch.sqrt((vy ** 2).sum(dim=1)) + 1e-8)
        return 1 - costheta.mean()

import logging


def train(args):
    # Logger & device
    logger = setup_logger(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    train_dataset = CortexDataset(args.data_root, args.subject, mode="train")
    test_dataset = CortexDataset(args.data_root, args.subject, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4)

    # Data stats for reference
    fmri_train = train_dataset.fmri_data
    logger.info(
        "Data Stats (train) - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f" % (
            fmri_train.mean().item(),
            fmri_train.std().item(),
            fmri_train.min().item(),
            fmri_train.max().item(),
        )
    )

    # Model: AE or VAE (with optional CLIP alignment)
    input_dim = fmri_train.shape[1]
    clip_dim = train_dataset.clip_data.shape[1]

    if args.model_type == "vae":
        model = BrainVAE(
            input_dim=input_dim,
            clip_dim=clip_dim,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
            kl_weight=args.kl_weight,
            clip_weight=args.clip_weight,
        ).to(device)
    else:
        model = BrainAutoencoder(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model Architecture: {model}")
    logger.info(f"Parameters: {n_params:.2f}M")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if args.early_stop_metric == "mse" else "max",
            factor=0.5,
            patience=args.scheduler_patience,
        )
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min,
        )

    # Loss
    criterion = nn.MSELoss()

    # Optional wandb
    if args.wandb:
        wandb.init(project="cortex-autoencoder", config=vars(args))
        wandb.watch(model)

    # Training Loop
    best_mse = float("inf")
    best_pearson = -float("inf")
    early_stop_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss_accum = 0.0
        grad_norm_accum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for fmri, clip in pbar:
            fmri = fmri.to(device)
            clip = clip.to(device)

            optimizer.zero_grad()

            if args.model_type == "vae":
                recon, z, rec_loss, kl_loss, clip_loss, total_loss = model(
                    fmri, clip, sample_posterior=True
                )
                loss = total_loss
                batch_train_mse = rec_loss.item()
            else:
                recon, z = model(fmri)
                loss = criterion(recon, fmri)
                batch_train_mse = loss.item()

            loss.backward()

            # Gradient norm for logging
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm_sq += param_norm ** 2
            total_norm = total_norm_sq ** 0.5

            optimizer.step()

            train_loss_accum += batch_train_mse
            grad_norm_accum += total_norm
            pbar.set_postfix({"loss": loss.item(), "grad": f"{total_norm:.3f}"})

        avg_train_loss = train_loss_accum / len(train_loader)
        avg_grad_norm = grad_norm_accum / len(train_loader)

        # Validation
        model.eval()
        test_loss_accum = 0.0
        test_pearson_accum = 0.0

        with torch.no_grad():
            for fmri, clip in test_loader:
                fmri = fmri.to(device)
                clip = clip.to(device)

                if args.model_type == "vae":
                    recon, z, _, _, _, _ = model(
                        fmri, clip, sample_posterior=False
                    )
                else:
                    recon, z = model(fmri)

                loss = criterion(recon, fmri)
                test_loss_accum += loss.item()

                # Pearson (batch-wise)
                vx = fmri - fmri.mean(dim=1, keepdim=True)
                vy = recon - recon.mean(dim=1, keepdim=True)
                costheta = (vx * vy).sum(dim=1) / (
                    torch.sqrt((vx ** 2).sum(dim=1)) *
                    torch.sqrt((vy ** 2).sum(dim=1)) + 1e-8
                )
                test_pearson_accum += costheta.mean().item()

        avg_test_loss = test_loss_accum / len(test_loader)
        avg_test_pearson = test_pearson_accum / len(test_loader)

        # Scheduler step
        if scheduler is not None:
            if args.scheduler == "plateau":
                val_metric = avg_test_loss if args.early_stop_metric == "mse" else avg_test_pearson
                scheduler.step(val_metric)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d: Train MSE: %.4f, GradNorm: %.4f, Test MSE: %.4f, Test Pearson: %.4f, LR: %.1e"
            % (epoch + 1, avg_train_loss, avg_grad_norm, avg_test_loss, avg_test_pearson, current_lr)
        )

        if args.wandb:
            wandb.log(
                {
                    "train_mse": avg_train_loss,
                    "test_mse": avg_test_loss,
                    "test_pearson": avg_test_pearson,
                    "grad_norm": avg_grad_norm,
                    "lr": current_lr,
                }
            )

        # Select metric for early stopping / checkpointing
        if args.early_stop_metric == "mse":
            current_metric = -avg_test_loss  # larger is better when negated
            best_metric = -best_mse
        else:
            current_metric = avg_test_pearson
            best_metric = best_pearson

        # Checkpointing based on chosen metric
        if current_metric > best_metric:
            if args.early_stop_metric == "mse":
                best_mse = avg_test_loss
            else:
                best_pearson = avg_test_pearson

            early_stop_counter = 0
            ckpt_path = os.path.join(args.save_dir, "best_ae.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved best model to {ckpt_path}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.patience:
            if args.no_early_stop:
                logger.info(
                    f"Early stop triggered at epoch {epoch+1} but disabled. Resetting counter."
                )
                early_stop_counter = 0
            else:
                logger.info("Early stopping triggered.")
                break

    # Save final
    last_path = os.path.join(args.save_dir, "last_ae.pth")
    torch.save(model.state_dict(), last_path)
    logger.info(f"Final model saved to {last_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/processed")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument(
        "--model_type",
        type=str,
        default="ae",
        choices=["ae", "vae"],
        help="Use standard autoencoder or VAE with KL + CLIP alignment",
    )
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum LR for cosine scheduler")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1e-4, help="Weight for VAE KL loss")
    parser.add_argument("--clip_weight", type=float, default=0.0, help="Weight for CLIP alignment loss")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)

    # Scheduler / early stopping
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["none", "plateau", "cosine"],
        help="LR scheduler type",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=10,
        help="Patience for ReduceLROnPlateau scheduler",
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="mse",
        choices=["mse", "pearson"],
        help="Metric used for early stopping and best checkpoint",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--save_dir", type=str, default="checkpoints/brain_ae")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no_early_stop", action="store_true", help="Disable early stopping")
    
    args = parser.parse_args()
    train(args)
