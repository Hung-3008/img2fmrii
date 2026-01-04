import torch
import sys
import os

# Ensure we can import from local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autoencoder import BrainAutoencoder

def test_model():
    print("Initializing model...")
    try:
        model = BrainAutoencoder(latent_dim=1024).cuda()
    except AssertionError as e:
        print(f"Initialization failed: {e}")
        return

    print("Model initialized.")
    x = torch.randn(4, 15724).cuda()
    print(f"Input shape: {x.shape}")
    
    try:
        recon, z = model(x)
        print(f"Recon shape: {recon.shape}")
        print(f"Latent shape: {z.shape}")
        assert recon.shape == x.shape
        assert z.shape == (4, 1024)
        print("Forward pass successful.")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e

if __name__ == "__main__":
    test_model()
