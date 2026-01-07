"""
Hybrid Prediction Module for CortexFlow

Combines regressor-based mean prediction with flow-based refinement
for improved accuracy on MSE and Pearson metrics.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class HybridPredictor(nn.Module):
    """
    Hybrid prediction strategy combining:
    1. Regressor: deterministic mean prediction from CLIP
    2. Flow: learned residual distribution for refinement
    
    Prediction modes:
    - 'mean_only': z = regressor(clip) - deterministic, low variance
    - 'flow_only': z = flow.inverse(noise, clip) - stochastic, high variance
    - 'hybrid': z = regressor(clip) + alpha * flow_residual - balanced
    - 'multi_sample': average multiple flow samples - reduced variance
    - 'residual': z = regressor(clip) + flow.inverse(noise, clip) - flow learns residual
    """
    
    def __init__(
        self,
        regressor: nn.Module,
        flow: nn.Module,
        decoder: nn.Module,
        alpha: float = 0.5,
        temperature: float = 0.8,
    ):
        """
        Args:
            regressor: MLP that maps CLIP -> latent mean
            flow: CortexFlow model for distribution learning
            decoder: Autoencoder decoder (latent -> fMRI)
            alpha: Blending factor for hybrid mode (0 = mean only, 1 = full flow)
            temperature: Sampling temperature for noise (< 1 reduces variance)
        """
        super().__init__()
        self.regressor = regressor
        self.flow = flow
        self.decoder = decoder
        self.alpha = alpha
        self.temperature = temperature
    
    def predict_latent(
        self,
        clip: torch.Tensor,
        mode: Literal['mean_only', 'flow_only', 'hybrid', 'multi_sample', 'residual'] = 'hybrid',
        num_samples: int = 10,
        return_all_samples: bool = False,
    ) -> torch.Tensor:
        """
        Predict latent representation from CLIP embedding.
        
        Args:
            clip: CLIP embeddings [B, clip_dim]
            mode: Prediction strategy
            num_samples: Number of samples for 'multi_sample' mode
            return_all_samples: If True, return all samples instead of mean
            
        Returns:
            z_pred: Predicted latent [B, latent_dim] or [num_samples, B, latent_dim]
        """
        B = clip.shape[0]
        device = clip.device
        
        # Get regressor mean
        z_mean = self.regressor(clip)  # [B, latent_dim]
        latent_dim = z_mean.shape[1]
        
        if mode == 'mean_only':
            return z_mean
        
        elif mode == 'flow_only':
            noise = torch.randn(B, latent_dim, device=device) * self.temperature
            z_pred = self.flow.inverse(noise, clip)
            return z_pred
        
        elif mode == 'hybrid':
            # Flow inverse with scaled noise, blend with mean
            noise = torch.randn(B, latent_dim, device=device) * self.temperature
            z_flow = self.flow.inverse(noise, clip)
            z_pred = (1 - self.alpha) * z_mean + self.alpha * z_flow
            return z_pred
        
        elif mode == 'residual':
            # Flow learns residual from mean
            noise = torch.randn(B, latent_dim, device=device) * self.temperature
            z_residual = self.flow.inverse(noise, clip)
            z_pred = z_mean + z_residual
            return z_pred
        
        elif mode == 'multi_sample':
            samples = []
            for _ in range(num_samples):
                noise = torch.randn(B, latent_dim, device=device) * self.temperature
                z_sample = self.flow.inverse(noise, clip)
                samples.append(z_sample)
            
            samples = torch.stack(samples)  # [num_samples, B, latent_dim]
            
            if return_all_samples:
                return samples
            else:
                return samples.mean(dim=0)  # [B, latent_dim]
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def predict_fmri(
        self,
        clip: torch.Tensor,
        mode: Literal['mean_only', 'flow_only', 'hybrid', 'multi_sample', 'residual'] = 'hybrid',
        num_samples: int = 10,
    ) -> torch.Tensor:
        """
        Predict fMRI from CLIP embedding (end-to-end).
        
        Args:
            clip: CLIP embeddings [B, clip_dim]
            mode: Prediction strategy
            num_samples: Number of samples for 'multi_sample' mode
            
        Returns:
            x_pred: Predicted fMRI [B, fmri_dim]
        """
        z_pred = self.predict_latent(clip, mode=mode, num_samples=num_samples)
        x_pred = self.decoder(z_pred)
        return x_pred
    
    def forward(
        self,
        clip: torch.Tensor,
        mode: str = 'hybrid',
        num_samples: int = 10,
    ) -> torch.Tensor:
        """Alias for predict_fmri."""
        return self.predict_fmri(clip, mode=mode, num_samples=num_samples)


class ResidualFlowTrainer:
    """
    Training wrapper for residual flow strategy.
    
    Instead of training flow to map z_target -> N(0,1),
    trains flow to map (z_target - z_regressor) -> N(0,1).
    
    This lets the regressor handle the mean prediction,
    while the flow captures the residual distribution.
    """
    
    def __init__(
        self,
        flow: nn.Module,
        regressor: nn.Module,
        detach_regressor: bool = True,
    ):
        """
        Args:
            flow: CortexFlow model
            regressor: Regressor model
            detach_regressor: If True, don't backprop through regressor for flow loss
        """
        self.flow = flow
        self.regressor = regressor
        self.detach_regressor = detach_regressor
    
    def compute_residual_nll(
        self,
        z_target: torch.Tensor,
        clip: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute NLL loss on residual distribution.
        
        Args:
            z_target: Target latent from autoencoder [B, latent_dim]
            clip: CLIP embeddings [B, clip_dim]
            
        Returns:
            nll_loss: Negative log-likelihood loss
            z_gauss: Transformed residual (should be ~N(0,1))
        """
        import numpy as np
        
        # Predict mean from regressor
        z_mean = self.regressor(clip)
        if self.detach_regressor:
            z_mean = z_mean.detach()
        
        # Compute residual
        z_residual = z_target - z_mean
        
        # Transform residual through flow
        z_gauss, log_det = self.flow(z_residual, clip)
        
        # Compute NLL per dimension
        D = z_gauss.shape[1]
        log_p_base = -0.5 * (z_gauss ** 2 + np.log(2 * np.pi)).sum(dim=1)
        nll_loss = -(log_p_base + log_det).mean() / D
        
        return nll_loss, z_gauss
