"""
CortexFlow Latent Training v2

Improved training script with:
- Residual flow training (flow learns z_target - z_regressor)
- Direct Pearson loss on decoded fMRI
- Multi-sample validation
- Better loss weighting
"""

import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset import CortexDataset
from autoencoder import BrainAutoencoder, BrainVAE
from model import CortexFlow
from loss import CortexFlowLoss, PearsonLoss
from hybrid_sampler import HybridPredictor, ResidualFlowTrainer


def setup_logger(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("CortexFlowV2")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_path = os.path.join(save_dir, "train_latent_v2.log")
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


def compute_pearson(pred, target):
    """Compute batch-wise Pearson correlation."""
    vx = pred - pred.mean(dim=1, keepdim=True)
    vy = target - target.mean(dim=1, keepdim=True)
    pearson = (vx * vy).sum(dim=1) / (
        torch.sqrt((vx**2).sum(dim=1)) * torch.sqrt((vy**2).sum(dim=1)) + 1e-8
    )
    return pearson.mean()


def train(args):
    logger = setup_logger(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Training mode: {args.train_mode}")
    logger.info(f"Flow mode: {'residual' if args.residual_flow else 'standard'}")
    
    # 1. Load Data
    train_dataset = CortexDataset(args.data_root, args.subject, mode="train")
    test_dataset = CortexDataset(args.data_root, args.subject, mode="test")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Check data dims
    sample_fmri, sample_clip = train_dataset[0]
    fmri_dim = sample_fmri.shape[0]
    clip_dim = sample_clip.shape[0]
    logger.info(f"Data dims: fMRI={fmri_dim}, CLIP={clip_dim}")
    
    # 2. Load Frozen Autoencoder / VAE
    logger.info("Loading Autoencoder...")
    if args.ae_type == "vae":
        ae = BrainVAE(
            input_dim=fmri_dim,
            clip_dim=clip_dim,
            latent_dim=args.latent_dim,
            dropout=0.1,
            kl_weight=0.0,
            clip_weight=0.0,
        ).to(device)
    else:
        ae = BrainAutoencoder(
            input_dim=fmri_dim,
            latent_dim=args.latent_dim
        ).to(device)
    
    if args.ae_ckpt and os.path.exists(args.ae_ckpt):
        try:
            state_dict = torch.load(args.ae_ckpt, map_location=device)
            ae.load_state_dict(state_dict)
            logger.info("Autoencoder weights loaded.")
        except Exception as e:
            logger.error(f"Error loading AE weights: {e}")
            return
    else:
        logger.warning(f"AE checkpoint {args.ae_ckpt} not found!")
        return
        
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # 3. Initialize Flow & Regressor
    logger.info("Initializing CortexFlow + Regressor...")
    flow = CortexFlow(
        x_dim=args.latent_dim,
        c_dim=clip_dim,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Deeper regressor with residual connections option
    regressor = nn.Sequential(
        nn.Linear(clip_dim, 2048),
        nn.LayerNorm(2048),
        nn.GELU(),
        nn.Dropout(args.reg_dropout),
        nn.Linear(2048, 1024),
        nn.LayerNorm(1024),
        nn.GELU(),
        nn.Dropout(args.reg_dropout),
        nn.Linear(1024, args.latent_dim)
    ).to(device)
    
    flow_params = sum(p.numel() for p in flow.parameters())
    reg_params = sum(p.numel() for p in regressor.parameters())
    logger.info(f"Flow Params: {flow_params/1e6:.2f}M, Regressor Params: {reg_params/1e6:.2f}M")
    
    # Initialize loss function
    loss_fn = CortexFlowLoss(
        nll_weight=args.nll_weight,
        mse_weight=args.mse_weight,
        cos_weight=args.cos_weight,
        pearson_weight=args.pearson_weight,
    )
    
    # Residual flow trainer (optional)
    residual_trainer = None
    if args.residual_flow:
        residual_trainer = ResidualFlowTrainer(
            flow=flow,
            regressor=regressor,
            detach_regressor=True,
        )
    
    # Build optimizer
    train_mode = args.train_mode
    optim_params = []
    if train_mode in ["joint", "flow_only"]:
        optim_params += list(flow.parameters())
    if train_mode in ["joint", "reg_only"]:
        optim_params += list(regressor.parameters())
    
    optimizer = optim.AdamW(
        optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 4. Training Loop
    best_metric = -1.0
    early_stop = 0
    
    for epoch in range(args.epochs):
        flow.train()
        regressor.train()
        
        train_losses = {'nll': 0, 'mse': 0, 'cosine': 0, 'pearson': 0, 'total': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for fmri, clip in pbar:
            fmri = fmri.to(device)
            clip = clip.to(device)
            
            with torch.no_grad():
                if args.ae_type == "vae":
                    _, z_target, _, _, _, _ = ae(fmri, clip=None, sample_posterior=False)
                else:
                    _, z_target = ae(fmri)
            
            optimizer.zero_grad()
            
            # Regressor prediction
            z_pred = regressor(clip)
            
            # Flow loss (residual or standard)
            if train_mode in ["joint", "flow_only"]:
                if args.residual_flow and residual_trainer is not None:
                    nll_loss, z_gauss = residual_trainer.compute_residual_nll(z_target, clip)
                else:
                    z_gauss, log_det = flow(z_target, clip)
                    D = z_gauss.shape[1]
                    log_p_base = -0.5 * (z_gauss ** 2 + np.log(2 * np.pi)).sum(dim=1)
                    nll_loss = -(log_p_base + log_det).mean() / D
            else:
                nll_loss = torch.zeros(1, device=device)
                z_gauss = None
            
            # Decode for Pearson loss
            x_pred = ae.decoder(z_pred)
            
            # Compute combined loss
            losses = loss_fn(
                z_pred=z_pred,
                z_target=z_target,
                x_pred=x_pred,
                x_target=fmri,
                z_gauss=z_gauss if train_mode in ["joint", "flow_only"] else None,
                log_det=log_det if (train_mode in ["joint", "flow_only"] and not args.residual_flow) else None,
            )
            
            # Override NLL if using residual flow
            if args.residual_flow and train_mode in ["joint", "flow_only"]:
                losses['nll'] = nll_loss
                losses['total'] = losses['total'] - loss_fn.nll_weight * losses.get('nll', 0) + loss_fn.nll_weight * nll_loss
            
            # Select loss based on training mode
            if train_mode == "joint":
                loss = losses['total']
            elif train_mode == "reg_only":
                loss = losses['mse'] * args.mse_weight + losses['cosine'] * args.cos_weight
                if args.pearson_weight > 0:
                    loss = loss + losses['pearson'] * args.pearson_weight
            else:  # flow_only
                loss = nll_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(flow.parameters()) + list(regressor.parameters()), 1.0)
            optimizer.step()
            
            # Accumulate losses
            for k in train_losses:
                if k in losses:
                    train_losses[k] += losses[k].item() if isinstance(losses[k], torch.Tensor) else losses[k]
            
            pbar.set_postfix({
                'nll': losses.get('nll', 0).item() if isinstance(losses.get('nll', 0), torch.Tensor) else 0,
                'mse': losses['mse'].item(),
                'pear': losses.get('pearson', 0).item() if isinstance(losses.get('pearson', 0), torch.Tensor) else 0,
            })
            
        scheduler.step()
        
        # Validation
        flow.eval()
        regressor.eval()
        
        test_metrics = {'nll': 0, 'p_reg': 0, 'p_flow': 0, 'p_hybrid': 0, 'mse': 0}
        
        with torch.no_grad():
            for fmri, clip in test_loader:
                fmri = fmri.to(device)
                clip = clip.to(device)
                
                if args.ae_type == "vae":
                    _, z_target, _, _, _, _ = ae(fmri, clip=None, sample_posterior=False)
                else:
                    _, z_target = ae(fmri)
                
                # Regressor prediction
                z_pred_reg = regressor(clip)
                x_pred_reg = ae.decoder(z_pred_reg)
                
                # Flow sampling
                z_noise = torch.randn_like(z_target)
                if args.residual_flow:
                    z_residual = flow.inverse(z_noise * args.temperature, clip)
                    z_pred_flow = z_pred_reg + z_residual
                else:
                    z_pred_flow = flow.inverse(z_noise * args.temperature, clip)
                x_pred_flow = ae.decoder(z_pred_flow)
                
                # Hybrid prediction
                z_pred_hybrid = 0.7 * z_pred_reg + 0.3 * z_pred_flow
                x_pred_hybrid = ae.decoder(z_pred_hybrid)
                
                # Compute metrics
                test_metrics['p_reg'] += compute_pearson(x_pred_reg, fmri).item()
                test_metrics['p_flow'] += compute_pearson(x_pred_flow, fmri).item()
                test_metrics['p_hybrid'] += compute_pearson(x_pred_hybrid, fmri).item()
                test_metrics['mse'] += nn.functional.mse_loss(x_pred_reg, fmri).item()
                
                # NLL
                if train_mode in ["joint", "flow_only"]:
                    if args.residual_flow:
                        z_residual_target = z_target - z_pred_reg
                        z_gauss, log_det = flow(z_residual_target, clip)
                    else:
                        z_gauss, log_det = flow(z_target, clip)
                    D = z_gauss.shape[1]
                    log_p_base = -0.5 * (z_gauss ** 2 + np.log(2 * np.pi)).sum(dim=1)
                    test_metrics['nll'] += (-(log_p_base + log_det).mean() / D).item()

        # Average metrics
        n_batches = len(test_loader)
        for k in test_metrics:
            test_metrics[k] /= n_batches
        
        logger.info(
            f"Epoch {epoch+1}: Train Loss: {train_losses['total']/len(train_loader):.3f}, "
            f"Test NLL: {test_metrics['nll']:.3f}, "
            f"P_Reg: {test_metrics['p_reg']:.4f}, "
            f"P_Flow: {test_metrics['p_flow']:.4f}, "
            f"P_Hybrid: {test_metrics['p_hybrid']:.4f}, "
            f"MSE: {test_metrics['mse']:.4f}"
        )
        
        # Choose metric for early stopping
        if args.residual_flow:
            current_metric = test_metrics['p_hybrid']
        elif train_mode == "flow_only":
            current_metric = test_metrics['p_flow']
        else:
            current_metric = test_metrics['p_reg']
        
        if current_metric > best_metric:
            best_metric = current_metric
            if train_mode in ["joint", "flow_only"]:
                torch.save(flow.state_dict(), os.path.join(args.save_dir, "best_flow.pth"))
            if train_mode in ["joint", "reg_only"]:
                torch.save(regressor.state_dict(), os.path.join(args.save_dir, "best_regressor.pth"))
            early_stop = 0
            logger.info(f"  -> New best: {best_metric:.4f}")
        else:
            early_stop += 1
            
        if early_stop >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"Training complete. Best metric: {best_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/processed")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--ae_ckpt", required=True, help="Path to best_ae.pth")
    parser.add_argument("--ae_type", type=str, default="ae", choices=["ae", "vae"])
    parser.add_argument("--train_mode", type=str, default="joint", 
                        choices=["joint", "reg_only", "flow_only"])
    
    # Model architecture
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--reg_dropout", type=float, default=0.4)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    
    # Loss weights
    parser.add_argument("--nll_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=20.0)
    parser.add_argument("--cos_weight", type=float, default=2.0)
    parser.add_argument("--pearson_weight", type=float, default=5.0)
    
    # New features
    parser.add_argument("--residual_flow", action="store_true",
                        help="Use residual flow training (flow learns z - z_reg)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for flow (< 1 reduces variance)")
    
    parser.add_argument("--save_dir", default="checkpoints/cortexflow_v2")
    
    args = parser.parse_args()
    train(args)
