"""
CortexFlow Loss Functions

Enhanced loss functions for training CortexFlow model with:
- NLL: Flow negative log-likelihood
- MSE: Latent space MSE
- Pearson: Differentiable Pearson correlation loss on decoded fMRI
- Cosine: Latent cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PearsonLoss(nn.Module):
    """
    Differentiable Pearson correlation loss.
    Returns 1 - pearson_correlation so that minimizing loss maximizes correlation.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Pearson correlation loss.
        
        Args:
            pred: Predicted tensor [B, D]
            target: Target tensor [B, D]
            
        Returns:
            Loss scalar (1 - mean_pearson)
        """
        # Center the tensors
        vx = pred - pred.mean(dim=1, keepdim=True)
        vy = target - target.mean(dim=1, keepdim=True)
        
        # Compute correlation
        numerator = (vx * vy).sum(dim=1)
        denominator = torch.sqrt((vx ** 2).sum(dim=1)) * torch.sqrt((vy ** 2).sum(dim=1)) + self.eps
        
        pearson = numerator / denominator
        
        # Return 1 - mean(pearson) so minimizing maximizes correlation
        return 1.0 - pearson.mean()


class CosineLoss(nn.Module):
    """Cosine similarity loss in latent space."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        return 1.0 - cosine_sim.mean()


class CortexFlowLoss(nn.Module):
    """
    Combined loss for CortexFlow training.
    
    Components:
    - nll_weight: Weight for flow NLL loss
    - mse_weight: Weight for latent MSE loss  
    - cos_weight: Weight for latent cosine loss
    - pearson_weight: Weight for decoded fMRI Pearson loss
    """
    
    def __init__(
        self,
        nll_weight: float = 1.0,
        mse_weight: float = 20.0,
        cos_weight: float = 2.0,
        pearson_weight: float = 5.0,
    ):
        super().__init__()
        self.nll_weight = nll_weight
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight
        self.pearson_weight = pearson_weight
        
        self.pearson_loss = PearsonLoss()
        self.cosine_loss = CosineLoss()
    
    def compute_nll(self, z_gauss: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
        """Compute NLL loss per dimension for better scaling."""
        D = z_gauss.shape[1]
        log_p_base = -0.5 * (z_gauss ** 2 + np.log(2 * np.pi)).sum(dim=1)
        nll = -(log_p_base + log_det).mean() / D
        return nll
    
    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        x_pred: torch.Tensor = None,
        x_target: torch.Tensor = None,
        z_gauss: torch.Tensor = None,
        log_det: torch.Tensor = None,
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            z_pred: Predicted latent [B, latent_dim]
            z_target: Target latent [B, latent_dim]
            x_pred: Decoded fMRI prediction [B, D] (optional, for Pearson)
            x_target: Target fMRI [B, D] (optional, for Pearson)
            z_gauss: Flow-transformed latent [B, latent_dim] (optional, for NLL)
            log_det: Log determinant from flow [B] (optional, for NLL)
            
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        total = 0.0
        
        # MSE loss on latent
        mse_loss = F.mse_loss(z_pred, z_target)
        losses['mse'] = mse_loss
        total = total + self.mse_weight * mse_loss
        
        # Cosine loss on latent
        cos_loss = self.cosine_loss(z_pred, z_target)
        losses['cosine'] = cos_loss
        total = total + self.cos_weight * cos_loss
        
        # NLL loss from flow
        if z_gauss is not None and log_det is not None:
            nll_loss = self.compute_nll(z_gauss, log_det)
            losses['nll'] = nll_loss
            total = total + self.nll_weight * nll_loss
        
        # Pearson loss on decoded fMRI
        if x_pred is not None and x_target is not None and self.pearson_weight > 0:
            pearson_loss = self.pearson_loss(x_pred, x_target)
            losses['pearson'] = pearson_loss
            total = total + self.pearson_weight * pearson_loss
        
        losses['total'] = total
        return losses
