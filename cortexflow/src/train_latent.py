
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
from model import CortexFlow


def setup_logger(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("CortexLatentFlow")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_path = os.path.join(save_dir, "train_latent.log")
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

def train(args):
    logger = setup_logger(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    train_dataset = CortexDataset(args.data_root, args.subject, mode="train")
    test_dataset = CortexDataset(args.data_root, args.subject, mode="test")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # Full batch for valid? typically batch_size is fine
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
        logger.warning(f"AE checkpoint {args.ae_ckpt} not found! Flow will train on garbage latents.")
        
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
    
    # Simple Regressor: CLIP -> Latent Mean (Brain Space)
    # Deeper to capture complex mapping, High Dropout to prevent overfitting
    regressor = nn.Sequential(
        nn.Linear(clip_dim, 2048),
        nn.LayerNorm(2048),
        nn.GELU(),
        nn.Dropout(0.5), # Increased dropout
        nn.Linear(2048, 1024),
        nn.LayerNorm(1024), 
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(1024, args.latent_dim)
    ).to(device)
    
    logger.info(f"Flow Params: {sum(p.numel() for p in flow.parameters())/1e6:.2f}M")

    # Optionally load a pre-trained regressor (for flow-only stage)
    if getattr(args, "train_mode", "joint") == "flow_only":
        if args.reg_ckpt is None or (not os.path.exists(args.reg_ckpt)):
            logger.error("Flow-only mode requires --reg_ckpt pointing to a trained regressor.")
            return
        try:
            reg_state = torch.load(args.reg_ckpt, map_location=device)
            regressor.load_state_dict(reg_state)
            logger.info(f"Regressor weights loaded from {args.reg_ckpt}.")
        except Exception as e:
            logger.error(f"Error loading regressor weights: {e}")
            return
        for p in regressor.parameters():
            p.requires_grad = False
        regressor.eval()
    
    # Build optimizer parameter list according to training mode
    train_mode = getattr(args, "train_mode", "joint")
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
    
    if args.wandb:
        wandb.init(project="cortex-latent-flow", config=args)
        
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 4. Training Loop
    best_test_pearson = -1.0
    early_stop = 0
    
    for epoch in range(args.epochs):
        flow.train()
        regressor.train()
        
        train_nll_acc = 0
        train_reg_acc = 0
        
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
            
            # --- Conditional Prior Strategy ---
            # 1. Regressor predicts the "Mean" of the Latent Code
            z_pred_mu = regressor(clip) 
            
            # 2. Flow transforms Brain Latent -> Z-space (Noise)
            # In all modes we keep a standard normal prior for z.

            # Regressor Loss (only if regressor is being trained)
            if train_mode in ["joint", "reg_only"]:
                mse_loss = nn.functional.mse_loss(z_pred_mu, z_target)
                z_pred_norm = nn.functional.normalize(z_pred_mu, dim=1)
                z_target_norm = nn.functional.normalize(z_target, dim=1)
                cosine_loss = 1 - (z_pred_norm * z_target_norm).sum(dim=1).mean()
            else:
                mse_loss = torch.zeros(1, device=device)
                cosine_loss = torch.zeros(1, device=device)
            
            # Flow Loss (NLL per-dimension for better scaling, only if flow is trained)
            if train_mode in ["joint", "flow_only"]:
                z_gauss, log_det = flow(z_target, clip)
                D = z_gauss.shape[1]
                log_p_base = -0.5 * (z_gauss ** 2 + np.log(2 * np.pi)).sum(dim=1)
                nll_loss = -(log_p_base + log_det).mean() / D
            else:
                nll_loss = torch.zeros(1, device=device)
            
            # Combine losses according to training mode
            if train_mode == "joint":
                loss = (
                    nll_loss
                    + args.reg_mse_weight * mse_loss
                    + args.reg_cos_weight * cosine_loss
                )
            elif train_mode == "reg_only":
                loss = (
                    args.reg_mse_weight * mse_loss
                    + args.reg_cos_weight * cosine_loss
                )
            else:  # flow_only
                loss = nll_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()
            
            train_nll_acc += nll_loss.item()
            train_reg_acc += mse_loss.item()
            pbar.set_postfix({'nll': nll_loss.item(), 'mse': mse_loss.item(), 'cos': cosine_loss.item()})
            
        scheduler.step()
        
        # Validation
        flow.eval()
        regressor.eval()
        
        test_nll_acc = 0
        test_pearson_flow = 0
        test_pearson_reg = 0
        
        with torch.no_grad():
            for fmri, clip in test_loader:
                fmri = fmri.to(device)
                clip = clip.to(device)
                
                if args.ae_type == "vae":
                    _, z_target, _, _, _, _ = ae(fmri, clip=None, sample_posterior=False)
                else:
                    _, z_target = ae(fmri)
                
                # NLL (per-dimension) if flow is active
                if train_mode in ["joint", "flow_only"]:
                    z_gauss, log_det = flow(z_target, clip)
                    D = z_gauss.shape[1]
                    log_p_base = -0.5 * (z_gauss ** 2 + np.log(2 * np.pi)).sum(dim=1)
                    test_nll_acc += (-(log_p_base + log_det).mean() / D).item()
                
                # Pearson Flow (Sampled) if flow is defined
                if train_mode in ["joint", "flow_only"]:
                    z_noise = torch.randn_like(z_target)
                    z_pred_flow = flow.inverse(z_noise, clip)
                    x_pred_flow = ae.decoder(z_pred_flow)
                else:
                    x_pred_flow = None
                
                # Pearson Regressor (Deterministic)
                z_pred_reg = regressor(clip)
                x_pred_reg = ae.decoder(z_pred_reg)
                
                # Eval
                vx = fmri - fmri.mean(dim=1, keepdim=True)
                
                if x_pred_flow is not None:
                    vy_flow = x_pred_flow - x_pred_flow.mean(dim=1, keepdim=True)
                    pearson_f = (vx * vy_flow).sum(dim=1) / (torch.sqrt((vx**2).sum(dim=1)) * torch.sqrt((vy_flow**2).sum(dim=1)) + 1e-8)
                    test_pearson_flow += pearson_f.mean().item()
                
                vy_reg = x_pred_reg - x_pred_reg.mean(dim=1, keepdim=True)
                pearson_r = (vx * vy_reg).sum(dim=1) / (torch.sqrt((vx**2).sum(dim=1)) * torch.sqrt((vy_reg**2).sum(dim=1)) + 1e-8)
                test_pearson_reg += pearson_r.mean().item()

        avg_test_nll = test_nll_acc / len(test_loader) if train_mode in ["joint", "flow_only"] else 0.0
        avg_p_flow = test_pearson_flow / len(test_loader) if train_mode in ["joint", "flow_only"] else 0.0
        avg_p_reg = test_pearson_reg / len(test_loader)

        logger.info(
            "Epoch %d: Train NLL: %.2f, Test NLL: %.2f, P_Flow: %.4f, P_Reg: %.4f"
            % (epoch + 1, train_nll_acc / len(train_loader), avg_test_nll, avg_p_flow, avg_p_reg)
        )
        
        if args.wandb:
            wandb.log({
                "test_nll": avg_test_nll,
                "test_pearson_flow": avg_p_flow,
                "test_pearson_reg": avg_p_reg
            })
            
        # Stop based on metric depending on training mode
        if train_mode == "flow_only":
            current_metric = avg_p_flow
        else:
            current_metric = avg_p_reg

        if current_metric > best_test_pearson:
            best_test_pearson = current_metric
            # Save checkpoints according to mode
            if train_mode in ["joint", "flow_only"]:
                torch.save(flow.state_dict(), os.path.join(args.save_dir, "best_flow.pth"))
            if train_mode in ["joint", "reg_only"]:
                torch.save(regressor.state_dict(), os.path.join(args.save_dir, "best_regressor.pth"))
            early_stop = 0
        else:
            early_stop += 1
            
        if early_stop >= args.patience:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/processed")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--ae_ckpt", required=True, help="Path to best_ae.pth")
    parser.add_argument(
        "--ae_type",
        type=str,
        default="ae",
        choices=["ae", "vae"],
        help="Type of encoder: standard autoencoder or VAE",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="joint",
        choices=["joint", "reg_only", "flow_only"],
        help="Training mode: joint flow+reg, regressor-only, or flow-only",
    )
    
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12) # Increased from 8
    parser.add_argument("--hidden_dim", type=int, default=1024) # Increased from 512
    parser.add_argument("--dropout", type=float, default=0.1) # Reduced from 0.2
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--reg_mse_weight", type=float, default=20.0, help="Weight for latent MSE loss")
    parser.add_argument("--reg_cos_weight", type=float, default=2.0, help="Weight for latent cosine loss")
    parser.add_argument("--reg_ckpt", type=str, default=None, help="Path to pretrained regressor for flow_only mode")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save_dir", default="checkpoints/latent_flow")
    parser.add_argument("--wandb", action="store_true")
    
    args = parser.parse_args()
    train(args)
