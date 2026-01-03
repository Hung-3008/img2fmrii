
import torch
import torch.nn as nn
import numpy as np

class AffineCoupling(nn.Module):
    def __init__(self, num_features, cond_dim, hidden_dim=1024, dropout=0.0):
        """
        Affine Coupling Layer for RealNVP flow.
        Splits features into two halves. Transforms one half based on the other + conditioning.
        
        Args:
            num_features (int): Total dimension of input (fMRI voxels).
            cond_dim (int): Dimension of conditional input (CLIP).
            hidden_dim (int): Hidden dimension for the internal MLP.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.num_features = num_features
        # Split features into roughly two halves
        self.split_idx = num_features // 2
        
        # Determine dimension of the half that is kept constant (input to NN)
        in_dim = self.split_idx
        # Determine dimension of the half that is transformed (output of NN)
        out_dim = num_features - self.split_idx
        
        # Neural Network to predict Scale (s) and Translation (t)
        # Input: [x1, c] -> Output: [s, t] for x2
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim * 2) # Outputs s and t
        )
        
        # Initialize last layer for stable initial flow (s close to 0, t close to 0)
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x, c, reverse=False):
        """
        Args:
            x: Input tensor (B, num_features)
            c: Conditional tensor (B, cond_dim)
            reverse: If True, run inverse transformation (Latent -> Data).
                     If False, run forward transformation (Data -> Latent).
        """
        x1 = x[:, :self.split_idx]
        x2 = x[:, self.split_idx:]
        
        # Concat condition to the fixed half
        # x1_c = [batch, split_idx + cond_dim]
        net_input = torch.cat([x1, c], dim=1)
        
        # Predict parameters
        params = self.net(net_input)
        s, t = params.chunk(2, dim=1)
        
        # Scaling constraint (often tanh) to prevent instability
        s = torch.tanh(s) 
        
        if not reverse:
            # Forward: Data -> Latent (Training)
            # y1 = x1
            # y2 = x2 * exp(s) + t
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            log_det = torch.sum(s, dim=1) # Sum log|J| = sum(s)
            y = torch.cat([y1, y2], dim=1)
            return y, log_det
        else:
            # Reverse: Latent -> Data (Sampling)
            # x1 = y1
            # x2 = (y2 - t) * exp(-s)
            y1 = x1 # In reverse, input x is actually y from previous layer 
            y2 = x2
            
            x1_out = y1
            x2_out = (y2 - t) * torch.exp(-s)
            
            # Note: We don't usually track log_det in sampling path
            x_out = torch.cat([x1_out, x2_out], dim=1)
            return x_out

class CortexFlow(nn.Module):
    def __init__(self, x_dim=15724, c_dim=768, num_layers=4, hidden_dim=1024, dropout=0.0):
        """
        Flat Conditional Normalizing Flow.
        Stacks AffineCoupling layers with Permutations.
        
        Args:
            x_dim: Dimension of fMRI data.
            c_dim: Dimension of CLIP embedding.
            num_layers: Number of flow steps (coupling layers).
            hidden_dim: Hidden dimension of MLPs.
            dropout: Dropout probability.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.x_dim = x_dim
        
        # Create alternating layers
        for i in range(num_layers):
            self.layers.append(AffineCoupling(x_dim, c_dim, hidden_dim, dropout))
            
        # Fixed Permutations to mix dimensions
        # Each layer will have a corresponding random permutation
        self.register_buffer('perm_indices', torch.randn(num_layers, x_dim).argsort(dim=1))
        
    def forward(self, x, c):
        """
        Forward Pass (Training): Maps Data x -> Latent z
        Returns:
            z: Latent variable (Gaussian-like if trained well)
            log_det_sum: Sum of log-determinants of Jacobian
        """
        log_det_sum = 0
        
        for i, layer in enumerate(self.layers):
            # Apply Permutation (Shuffle inputs before coupling)
            # This ensures different variables get updated in different layers
            perm = self.perm_indices[i]
            x = x[:, perm]
            
            # Apply Coupling
            x, log_det = layer(x, c, reverse=False)
            log_det_sum = log_det_sum + log_det
            
        return x, log_det_sum # x is now z
        
    def inverse(self, z, c):
        """
        Inverse Pass (Sampling): Maps Latent z -> Data x
        """
        x = z
        
        # Iterate backwards
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            perm = self.perm_indices[i]
            inv_perm = torch.argsort(perm) # Inverse permutation index
            
            # Inverse Coupling
            x = layer(x, c, reverse=True)
            
            # Inverse Permutation
            x = x[:, inv_perm]
            
        return x
