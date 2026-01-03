
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import logging

from dataset import CortexDataset
from model import CortexFlow

def get_args():
    parser = argparse.ArgumentParser(description="Train CortexFlow SP002")
    parser.add_argument("--data_root", default="data/processed", help="Path to processed data")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_layers", type=int, default=8, help="Number of flow coupling layers")
    parser.add_argument("--hidden_dim", type=int, default=512, help="MLP hidden dimension (Reduced from 1024)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Optimizer weight decay")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="checkpoints", help="Directory to save models")
    parser.add_argument("--save_interval", type=int, default=10, help="Save every N epochs")
    return parser.parse_args()

def setup_logger(save_dir):
    logger = logging.getLogger("CortexFlow")
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File Handler
    fh = logging.FileHandler(os.path.join(save_dir, "training.log"))
    fh.setLevel(logging.INFO)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def nll_loss(z, log_det):
    """
    Negative Log Likelihood Loss.
    Normalized by dimension D for better readability.
    """
    # z: [Batch, D]
    D = z.shape[1]
    
    # Log prob of standard normal: -0.5 * (z^2 + log(2pi))
    const = 0.5 * np.log(2 * np.pi)
    prior_log_prob = -0.5 * torch.sum(z**2, dim=1) - D * const
    
    log_likelihood = prior_log_prob + log_det
    
    # Mean over batch, divided by D (bits/dim or nats/dim equivalent)
    loss = -torch.mean(log_likelihood) / D
    return loss

def train(args):
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(args.save_dir)
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Regularization: Dropout={args.dropout}, Weight Decay={args.weight_decay}, Patience={args.patience}")

    # Data
    train_ds = CortexDataset(args.data_root, args.subject, mode="train")
    test_ds = CortexDataset(args.data_root, args.subject, mode="test")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    x_dim = train_ds.fmri_data.shape[1]
    c_dim = train_ds.clip_data.shape[1]
    
    model = CortexFlow(
        x_dim=x_dim, 
        c_dim=c_dim, 
        num_layers=args.num_layers, 
        hidden_dim=args.hidden_dim, 
        dropout=args.dropout
    )
    model.to(device)
    logger.info(f"Model initialized: {args.num_layers} layers, {args.hidden_dim} hidden, {args.dropout} dropout. x_dim={x_dim}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_test_loss = float('inf')
    patience_counter = 0 # Early stopping counter
    
    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0
        
        # Use simple wrapper for progress bar to avoid cluttering log
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, c in pbar:
            x, c = x.to(device), c.to(device)
            
            optimizer.zero_grad()
            z, log_det = model(x, c)
            loss = nll_loss(z, log_det)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("Loss is NaN/Inf! Skipping batch.")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            pbar.set_postfix({"nll_per_dim": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # Validation
        model.eval()
        test_loss_sum = 0
        with torch.no_grad():
            for x, c in test_loader:
                x, c = x.to(device), c.to(device)
                z, log_det = model(x, c)
                loss = nll_loss(z, log_det)
                test_loss_sum += loss.item()
        
        avg_test_loss = test_loss_sum / len(test_loader)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f"Epoch {epoch}: Train NLL={avg_train_loss:.4f}, Test NLL={avg_test_loss:.4f}, LR={current_lr:.6f}")
        
        # Save Best & Early Stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0 # Reset counter
            
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"  New best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early Stopping Triggered: Test NLL has not improved for {args.patience} epochs.")
                logger.info(f"Best Test NLL was: {best_test_loss:.4f}")
                break
            
        # Interval Save
        if epoch % args.save_interval == 0:
             chk_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
             torch.save(model.state_dict(), chk_path)
             logger.info(f"  Checkpoint saved to {chk_path}")

if __name__ == "__main__":
    args = get_args()
    train(args)
