"""
CortexFlow → MindEye2 Image Reconstruction

Reconstructs 2D images from CortexFlow's synthetic fMRI using MindEye2.

Memory-optimized for 24GB VRAM:
- Sequential model loading (CortexFlow first, then MindEye2)
- FP16 inference
- Single-sample processing
- CUDA cache clearing between stages

Output: Comparison visualization
  Original Image | Real fMRI | Synthetic fMRI (CortexFlow) | Reconstructed Image (MindEye2)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import gc

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MINDEYE_SRC = os.path.join(SCRIPT_DIR, "../../MindEyeV2/src")
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "../..")

sys.path.insert(0, MINDEYE_SRC)
sys.path.insert(0, os.path.join(MINDEYE_SRC, "generative_models"))


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct images from CortexFlow synthetic fMRI")
    
    # Data paths
    parser.add_argument("--data_root", default="data/processed", help="Path to processed data")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument("--nsd_path", default="data/NSD", help="Path to NSD data (for images)")
    
    # CortexFlow checkpoints
    parser.add_argument("--ae_ckpt", default="checkpoints/cortexflow_vae/best_ae.pth")
    parser.add_argument("--flow_ckpt", default="checkpoints/cortexflow_v/best_flow.pth")
    parser.add_argument("--regressor_ckpt", default="checkpoints/cortexflow_v/best_regressor.pth")
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    
    # MindEye2 checkpoints
    parser.add_argument("--mindeye_ckpt", default="checkpoints/train_logs/final_subj01_pretrained_40sess_24bs")
    parser.add_argument("--unclip_ckpt", default="checkpoints/unclip6_epoch0_step110000.ckpt")
    parser.add_argument("--mindeye_hidden_dim", type=int, default=4096)
    parser.add_argument("--n_blocks", type=int, default=4)
    
    # Processing
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test samples to process")
    parser.add_argument("--sample_indices", type=int, nargs='+', default=None, 
                        help="Specific sample indices to process")
    parser.add_argument("--mode", default="mean_only", choices=["mean_only", "hybrid", "residual"],
                        help="CortexFlow prediction mode")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Output
    parser.add_argument("--output_dir", default="outputs/reconstructions")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def load_cortexflow(args, fmri_dim, clip_dim, device):
    """Load CortexFlow components (Autoencoder + Flow + Regressor)"""
    print("Loading CortexFlow components...")
    
    from autoencoder import BrainAutoencoder, BrainVAE
    from model import CortexFlow
    
    # Detect checkpoint type and load appropriate model
    ae_path = os.path.join(PROJECT_ROOT, args.ae_ckpt)
    if os.path.exists(ae_path):
        state_dict = torch.load(ae_path, map_location=device)
        
        # Check if it's a VAE (has fc_mu) or plain AE
        is_vae = 'fc_mu.weight' in state_dict
        
        if is_vae:
            print("  Detected VAE checkpoint")
            ae = BrainVAE(
                input_dim=fmri_dim,
                clip_dim=clip_dim,
                latent_dim=args.latent_dim,
                dropout=0.1,
                kl_weight=0.0,
                clip_weight=0.0,  # We don't need CLIP alignment for inference
            ).to(device)
            
            # Filter out clip_proj weights which may have different dimensions
            # (not needed for inference anyway since we only use encoder/decoder)
            filtered_state = {k: v for k, v in state_dict.items() 
                             if 'clip_proj' not in k}
            
            # Load with strict=False to handle any other potential mismatches
            missing, unexpected = ae.load_state_dict(filtered_state, strict=False)
            if missing:
                ignored = [k for k in missing if 'clip_proj' in k]
                real_missing = [k for k in missing if 'clip_proj' not in k]
                if ignored:
                    print(f"  Ignored {len(ignored)} clip_proj keys (not needed for inference)")
                if real_missing:
                    print(f"  Warning: Missing {len(real_missing)} keys: {real_missing[:3]}...")
        else:
            print("  Detected AE checkpoint")
            ae = BrainAutoencoder(
                input_dim=fmri_dim,
                latent_dim=args.latent_dim
            ).to(device)
            ae.load_state_dict(state_dict)
        
        print(f"  Loaded: {ae_path}")
        del state_dict
    else:
        print(f"  WARNING: AE not found at {ae_path}")
        ae = BrainAutoencoder(input_dim=fmri_dim, latent_dim=args.latent_dim).to(device)
    
    ae.eval()
    
    # Flow
    flow = CortexFlow(
        x_dim=args.latent_dim,
        c_dim=clip_dim,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=0.0
    ).to(device)
    
    flow_path = os.path.join(PROJECT_ROOT, args.flow_ckpt)
    if os.path.exists(flow_path):
        flow.load_state_dict(torch.load(flow_path, map_location=device))
        print(f"  Loaded Flow: {flow_path}")
    else:
        print(f"  WARNING: Flow not found at {flow_path}")
    
    flow.eval()
    
    # Regressor
    regressor = nn.Sequential(
        nn.Linear(clip_dim, 2048),
        nn.LayerNorm(2048),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(2048, 1024),
        nn.LayerNorm(1024),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(1024, args.latent_dim)
    ).to(device)
    
    reg_path = os.path.join(PROJECT_ROOT, args.regressor_ckpt)
    if os.path.exists(reg_path):
        regressor.load_state_dict(torch.load(reg_path, map_location=device))
        print(f"  Loaded Regressor: {reg_path}")
    else:
        print(f"  WARNING: Regressor not found at {reg_path}")
    
    regressor.eval()
    
    return ae, flow, regressor


def generate_synthetic_fmri(clip, ae, flow, regressor, mode="mean_only", temperature=0.8):
    """Generate synthetic fMRI from CLIP embedding using CortexFlow"""
    with torch.no_grad():
        # Get regressor prediction (mean)
        z_pred = regressor(clip)
        
        if mode == "mean_only":
            z_final = z_pred
        elif mode == "hybrid":
            # Blend with flow sample
            noise = torch.randn_like(z_pred) * temperature
            z_flow = flow.inverse(noise, clip)
            z_final = 0.7 * z_pred + 0.3 * z_flow
        elif mode == "residual":
            # Residual flow
            noise = torch.randn_like(z_pred) * temperature
            z_residual = flow.inverse(noise, clip)
            z_final = z_pred + z_residual
        else:
            z_final = z_pred
        
        # Decode to fMRI space
        fmri_synthetic = ae.decoder(z_final)
    
    return fmri_synthetic


def load_mindeye2(args, num_voxels, device):
    """Load MindEye2 components with memory optimization"""
    print("\nLoading MindEye2 components...")
    
    # Clear CUDA cache first
    torch.cuda.empty_cache()
    gc.collect()
    
    # Import MindEye2 modules
    from models import BrainNetwork, PriorNetwork, BrainDiffusionPrior
    import utils
    
    # RidgeRegression is defined inline in MindEye2 notebooks, so we define it here
    class RidgeRegression(nn.Module):
        def __init__(self, input_sizes, out_features):
            super(RidgeRegression, self).__init__()
            self.out_features = out_features
            self.linears = nn.ModuleList([
                nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
        def forward(self, x, subj_idx):
            out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
            return out
    
    clip_seq_dim = 256
    clip_emb_dim = 1664
    hidden_dim = args.mindeye_hidden_dim
    
    # Build model
    class MindEyeModule(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return x
    
    model = MindEyeModule()
    
    # Ridge Regression
    model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)
    
    # Brain Network (backbone)
    model.backbone = BrainNetwork(
        h=hidden_dim, 
        in_dim=hidden_dim, 
        seq_len=1,
        clip_size=clip_emb_dim, 
        out_dim=clip_emb_dim * clip_seq_dim,
        blurry_recon=False  # Disable to save memory
    )
    
    # Diffusion Prior
    out_dim = clip_emb_dim
    depth = 6
    dim_head = 52
    heads = clip_emb_dim // 52
    timesteps = 100
    
    prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=clip_seq_dim,
        learned_query_mode="pos_emb"
    )
    
    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
    
    # Load checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, args.mindeye_ckpt, "last.pth")
    print(f"  Loading MindEye2 checkpoint: {ckpt_path}")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Filter out blurry_recon related keys
        filtered_state = {k: v for k, v in state_dict.items() 
                         if 'blurry' not in k.lower() and 'blur' not in k.lower()}
        
        model.load_state_dict(filtered_state, strict=False)
        del checkpoint, state_dict
        print("  MindEye2 checkpoint loaded!")
    except Exception as e:
        print(f"  Error loading MindEye2: {e}")
        # Try DeepSpeed format
        try:
            import deepspeed
            outdir = os.path.join(PROJECT_ROOT, args.mindeye_ckpt)
            state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
                checkpoint_dir=outdir, tag='last'
            )
            filtered_state = {k: v for k, v in state_dict.items() 
                             if 'blurry' not in k.lower() and 'blur' not in k.lower()}
            model.load_state_dict(filtered_state, strict=False)
            del state_dict
            print("  Loaded from DeepSpeed format!")
        except Exception as e2:
            print(f"  DeepSpeed load also failed: {e2}")
    
    model.to(device)
    model.eval()
    
    return model, clip_seq_dim, clip_emb_dim


def load_unclip(args, device):
    """Load SDXL UnClip for image reconstruction - VRAM optimized"""
    print("\nLoading SDXL UnClip...")
    
    # Aggressive cache cleanup
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    
    from omegaconf import OmegaConf
    from generative_models.sgm.models.diffusion import DiffusionEngine
    
    # Load config
    config_path = os.path.join(MINDEYE_SRC, "generative_models/configs/unclip6.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    
    unclip_params = config["model"]["params"]
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    conditioner_config = unclip_params["conditioner_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
    
    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    sampler_config['params']['num_steps'] = 38
    
    diffusion_engine = DiffusionEngine(
        network_config=network_config,
        denoiser_config=denoiser_config,
        first_stage_config=first_stage_config,
        conditioner_config=conditioner_config,
        sampler_config=sampler_config,
        scale_factor=scale_factor,
        disable_first_stage_autocast=disable_first_stage_autocast
    )
    
    diffusion_engine.eval().requires_grad_(False)
    
    # Load checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, args.unclip_ckpt)
    print(f"  Loading UnClip checkpoint: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    diffusion_engine.load_state_dict(ckpt['state_dict'])
    del ckpt
    gc.collect()
    print("  Converting to BF16 for VRAM optimization...")
    diffusion_engine = diffusion_engine.to(dtype=torch.bfloat16)
    
    # Clear cache before moving to GPU
    torch.cuda.empty_cache()
    gc.collect()
    
    diffusion_engine.to(device)
    
    # Get vector suffix
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # Use proper size 224x224
        batch = {
            "jpg": torch.randn(1, 3, 224, 224, dtype=torch.bfloat16).to(device),
            "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
            "crop_coords_top_left": torch.zeros(1, 2).to(device)
        }
        out = diffusion_engine.conditioner(batch)
        vector_suffix = out["vector"].to(device)
        print(f"  [DEBUG] Vector Suffix: min={vector_suffix.min().item():.4f}, max={vector_suffix.max().item():.4f}, mean={vector_suffix.mean().item():.4f}")
    
    print("  UnClip loaded!")
    return diffusion_engine, vector_suffix


# Imoprt needed for unclip_recon
from generative_models.sgm.util import append_dims

def unclip_recon(x, diffusion_engine, vector_suffix, num_samples=1, offset_noise_level=0.04):
    """Local version of unclip_recon with explicit BF16 support for VRAM optimization"""
    device = x.device
    
    assert x.ndim==3
    if x.shape[0]==1:
        x = x[[0]]
    
    # Ensure inputs are bifloat16
    x = x.to(dtype=torch.bfloat16)
    vector_suffix = vector_suffix.to(dtype=torch.bfloat16)
        
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16), diffusion_engine.ema_scope():
        # Make noise explicitly bf16
        z = torch.randn(num_samples,4,96,96, dtype=torch.bfloat16).to(device) 

        token_shape = x.shape
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        # Make noise explicitly bf16
        tokens = torch.randn_like(x, dtype=torch.bfloat16)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z, dtype=torch.bfloat16)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device).to(dtype=torch.bfloat16)  # Ensure sigma is bf16
        print(f"  [DEBUG] Sigma[0]: {sigma.item():.4f}")

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device, dtype=torch.bfloat16), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim).to(z.device)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        ).to(z.device)
        
        # Ensure noised_z is bf16
        noised_z = noised_z.to(dtype=torch.bfloat16)

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(
                diffusion_engine.model, x.to(dtype=torch.bfloat16), sigma.to(dtype=torch.bfloat16), c
            )

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)
        return samples


def reconstruct_image(voxels, mindeye_model, diffusion_engine, vector_suffix, device):
    """Reconstruct image from fMRI voxels using MindEye2"""
    sys.path.insert(0, MINDEYE_SRC)
    import utils
    # MindEye2 Inference
    # Ensure inputs are bf16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # Ensure correct shape [B, seq, D]
        if voxels.dim() == 1:
            voxels = voxels.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        elif voxels.dim() == 2:
            voxels = voxels.unsqueeze(1)  # [B, 1, D]
        
        voxels = voxels.to(device).to(dtype=torch.bfloat16)  # Convert to BF16
        print(f"  [DEBUG] Input Voxels: min={voxels.min().item():.4f}, max={voxels.max().item():.4f}, mean={voxels.mean().item():.4f}, std={voxels.std().item():.4f}")
        
        # Ridge Regression
        voxel_ridge = mindeye_model.ridge(voxels, 0)  # [B, 1, hidden_dim]
        print(f"  [DEBUG] Ridge Out: min={voxel_ridge.min().item():.4f}, max={voxel_ridge.max().item():.4f}, mean={voxel_ridge.mean().item():.4f}")
        
        # Brain Network (runs in BF16 via autocast)
        backbone, clip_voxels, blurry_image_enc = mindeye_model.backbone(voxel_ridge)
        print(f"  [DEBUG] Backbone Out: min={backbone.min().item():.4f}, max={backbone.max().item():.4f}, mean={backbone.mean().item():.4f}")
        
        # Diffusion Prior
        prior_out = mindeye_model.diffusion_prior.p_sample_loop(
            backbone.shape,
            text_cond=dict(text_embed=backbone),
            cond_scale=1.,
            timesteps=20
        )
        print(f"  [DEBUG] Prior Out: min={prior_out.min().item():.4f}, max={prior_out.max().item():.4f}, mean={prior_out.mean().item():.4f}")
        
        # Check/Normalize Prior Out
        prior_norm = prior_out.norm(dim=-1, keepdim=True)
        print(f"  [DEBUG] Prior Norm: min={prior_norm.min().item():.4f}, max={prior_norm.max().item():.4f}, mean={prior_norm.mean().item():.4f}")
        
        # CLIP embeddings should be normalized?
        # Let's normalize it to be safe for UnClip
        prior_out = prior_out / (prior_norm + 1e-6)
        print(f"  [DEBUG] Normalized Prior Out: min={prior_out.min().item():.4f}, max={prior_out.max().item():.4f}")
        
        # UnClip reconstruction
        # Use local unclip_recon with explicit FP16
        samples = unclip_recon(
            prior_out,
            diffusion_engine,
            vector_suffix,
            num_samples=1
        )
        print(f"  [DEBUG] UnClip Samples: min={samples.min().item():.4f}, max={samples.max().item():.4f}, mean={samples.mean().item():.4f}")
        
    return samples


def create_fmri_lineplot(fmri, title="fMRI", ylim=None):
    """Create a 1D line plot visualization of fMRI data"""
    import io
    
    data = fmri.flatten()
    
    fig, ax = plt.subplots(figsize=(6, 2))  # Wider aspect ratio for time series/vector
    ax.plot(data, linewidth=0.5, color='black', alpha=0.8)
    ax.set_title(title, fontsize=10)
    
    # Remove x-axis labels as they are just voxel indices
    ax.set_xticks([])
    
    # Set y-limits if provided for consistent comparison
    if ylim:
        ax.set_ylim(ylim)
        
    # Add minimal grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Convert to PIL Image using BytesIO
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img.convert('RGB')


def load_test_data(args, project_root):
    """Load test data: fMRI, CLIP, and image IDs"""
    data_dir = os.path.join(project_root, args.data_root, args.subject)
    
    # Load test fMRI
    fmri_path = os.path.join(data_dir, "test_fmri_avg.npy")
    if os.path.exists(fmri_path):
        test_fmri = np.load(fmri_path).astype(np.float32)
        print(f"Loaded test fMRI: {test_fmri.shape}")
    else:
        raise FileNotFoundError(f"Test fMRI not found: {fmri_path}")
    
    # Load test CLIP - prioritize vitl14 (768-dim) which matches CortexFlow training
    clip_path = os.path.join(data_dir, "test_clip_vitl14.npy")
    if not os.path.exists(clip_path):
        # Fallback to multilayer if vitl14 not available
        clip_path = os.path.join(data_dir, "test_clip_multilayer.npy")
    
    if os.path.exists(clip_path):
        test_clip = np.load(clip_path).astype(np.float32)
        print(f"Loaded test CLIP: {test_clip.shape} from {os.path.basename(clip_path)}")
    else:
        raise FileNotFoundError(f"Test CLIP not found")
    
    # Load image IDs
    imgid_path = os.path.join(data_dir, "test_imgIds.npy")
    if os.path.exists(imgid_path):
        test_imgids = np.load(imgid_path)
        print(f"Loaded image IDs: {test_imgids.shape}")
    else:
        test_imgids = np.arange(len(test_fmri))
        print("Using sequential image IDs")
    
    return test_fmri, test_clip, test_imgids


def load_nsd_image(img_id, nsd_path, project_root):
    """Load original NSD image by ID"""
    # Try HDF5 format first
    hdf5_path = os.path.join(project_root, nsd_path, "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
    
    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            if 'imgBrick' in f:
                img = f['imgBrick'][int(img_id)]
                return Image.fromarray(img)
    
    # Fallback: try COCO images
    coco_path = os.path.join(project_root, nsd_path, "coco_images_224_float16.hdf5")
    if os.path.exists(coco_path):
        with h5py.File(coco_path, 'r') as f:
            if 'images' in f:
                img = f['images'][int(img_id)]
                if img.dtype == np.float16:
                    img = (img * 255).astype(np.uint8)
                return Image.fromarray(img)
    
    # Create placeholder
    print(f"  Warning: Could not load image {img_id}")
    placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
    placeholder[:, :, 2] = 50  # Dark blue
    return Image.fromarray(placeholder)


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load test data
    print("\n" + "="*60)
    print("Loading test data...")
    print("="*60)
    test_fmri, test_clip, test_imgids = load_test_data(args, PROJECT_ROOT)
    
    fmri_dim = test_fmri.shape[1]
    clip_dim = test_clip.shape[1]
    print(f"fMRI dim: {fmri_dim}, CLIP dim: {clip_dim}")
    
    # Load CortexFlow
    print("\n" + "="*60)
    print("Stage 1: Loading CortexFlow")
    print("="*60)
    ae, flow, regressor = load_cortexflow(args, fmri_dim, clip_dim, device)
    
    # Determine which samples to process
    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    else:
        n_samples = min(args.num_samples, len(test_fmri))
        sample_indices = np.random.choice(len(test_fmri), n_samples, replace=False)
    
    print(f"\nProcessing samples: {sample_indices}")
    
    # Generate synthetic fMRI for selected samples
    print("\nGenerating synthetic fMRI with CortexFlow...")
    synthetic_fmri_list = []
    
    for idx in tqdm(sample_indices, desc="CortexFlow"):
        clip_tensor = torch.from_numpy(test_clip[idx:idx+1]).to(device)
        syn_fmri = generate_synthetic_fmri(clip_tensor, ae, flow, regressor, mode=args.mode)
        synthetic_fmri_list.append(syn_fmri.cpu().numpy())
    
    synthetic_fmri = np.concatenate(synthetic_fmri_list, axis=0)
    print(f"Generated synthetic fMRI: {synthetic_fmri.shape}")
    
    # Free CortexFlow memory
    del ae, flow, regressor
    torch.cuda.empty_cache()
    gc.collect()
    print("CortexFlow unloaded, CUDA cache cleared")
    
    # Load MindEye2
    print("\n" + "="*60)
    print("Stage 2: Loading MindEye2")
    print("="*60)
    mindeye_model, clip_seq_dim, clip_emb_dim = load_mindeye2(args, fmri_dim, device)
    
    # Load UnClip
    diffusion_engine, vector_suffix = load_unclip(args, device)
    
    # Reconstruct images
    print("\n" + "="*60)
    print("Stage 3: Reconstructing images with MindEye2")
    print("="*60)
    
    reconstructed_images = []
    
    for i, idx in enumerate(tqdm(sample_indices, desc="MindEye2 Reconstruction")):
        syn_voxels = torch.from_numpy(synthetic_fmri[i:i+1])
        
        try:
            recon = reconstruct_image(
                syn_voxels, 
                mindeye_model, 
                diffusion_engine, 
                vector_suffix, 
                device
            )
            reconstructed_images.append(recon.cpu())
        except Exception as e:
            print(f"  Error reconstructing sample {idx}: {e}")
            # Create placeholder
            reconstructed_images.append(torch.zeros(1, 3, 768, 768))
        
        # Clear cache after each sample
        torch.cuda.empty_cache()
    
    # Create comparison visualizations
    print("\n" + "="*60)
    print("Stage 4: Creating comparison visualizations")
    print("="*60)
    
    from torchvision import transforms
    
    # Determine global min/max for fMRI plotting to keep scales consistent
    # Use 1st/99th percentiles to avoid outliers squashing the plot
    all_real = test_fmri[sample_indices]
    all_syn = synthetic_fmri
    
    # Calculate robust limits
    vmin = min(np.percentile(all_real, 1), np.percentile(all_syn, 1))
    vmax = max(np.percentile(all_real, 99), np.percentile(all_syn, 99))
    # Add some padding
    range_span = vmax - vmin
    ylim = (vmin - 0.1 * range_span, vmax + 0.1 * range_span)
    
    print(f"Plotting fMRI with y-limits: {ylim}")
    
    for i, idx in enumerate(sample_indices):
        print(f"Creating visualization for sample {idx}...")
        
        # Load original image
        img_id = test_imgids[idx] if idx < len(test_imgids) else idx
        original_img = load_nsd_image(img_id, args.nsd_path, PROJECT_ROOT)
        original_img = original_img.resize((512, 512)) # Larger for better visibility
        
        # Create fMRI line plots
        real_fmri_img = create_fmri_lineplot(test_fmri[idx], "Real fMRI", ylim)
        syn_fmri_img = create_fmri_lineplot(synthetic_fmri[i], "Synthetic fMRI", ylim)
        
        # Get reconstructed image
        if i < len(reconstructed_images):
            # Convert outputs
            recon_tensor = reconstructed_images[i][0]  # [3, H, W]
            if recon_tensor.dtype == torch.bfloat16:
                recon_tensor = recon_tensor.float()
            recon_img = transforms.ToPILImage()(recon_tensor)
            recon_img = recon_img.resize((512, 512))
        else:
            recon_img = Image.new('RGB', (512, 512), color='gray')
        
        # Create comparison figure
        # Layout: 1 row, 4 columns
        # Images square, Line plots rectangular
        
        fig = plt.figure(figsize=(20, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.5, 1.5, 1])
        
        # 1. Original Image
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(original_img)
        ax0.set_title('Original Image', fontsize=12, pad=10)
        ax0.axis('off')
        
        # 2. Real fMRI Line Plot
        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(real_fmri_img)
        ax1.set_title('Real fMRI', fontsize=12, pad=10)
        ax1.axis('off')
        
        # 3. Synthetic fMRI Line Plot
        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(syn_fmri_img)
        ax2.set_title('Synthetic fMRI\n(CortexFlow)', fontsize=12, pad=10)
        ax2.axis('off')
        
        # 4. Reconstructed Image
        ax3 = fig.add_subplot(gs[3])
        ax3.imshow(recon_img)
        ax3.set_title('Reconstructed Image\n(MindEye2)', fontsize=12, pad=10)
        ax3.axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(args.output_dir, f"comparison_sample_{idx:04d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path}")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print(f"Output directory: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
