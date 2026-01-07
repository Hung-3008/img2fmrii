"""Real fMRI to MindEye2 Image Reconstruction.

Reconstructs 2D images directly from real fMRI using MindEye2.
Default model: `final_subj01_pretrained_40sess_24bs` (single-subject, 4096-dim).
Config: hidden_dim=4096, blurry_recon enabled by default, BF16, sequential
loading. Memory-optimized for limited VRAM (e.g., 24GB or less), but the
4096-dim backbone still requires more memory than the 1024-dim multisubject
model.
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
import io
from torchvision import transforms

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MINDEYE_SRC = os.path.join(SCRIPT_DIR, "../../MindEyeV2/src")
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "../..")

sys.path.insert(0, MINDEYE_SRC)
sys.path.insert(0, os.path.join(MINDEYE_SRC, "generative_models"))


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct images from Real fMRI (MindEye2 4096-dim subj01)")
    
    # Data paths
    parser.add_argument("--data_root", default="data/processed", help="Path to processed data")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument("--nsd_path", default="data/NSD", help="Path to NSD data (for images)")
    
    # MindEye2 checkpoints
    # Default to single-subject 4096-dim model for subj01
    parser.add_argument("--mindeye_ckpt", default="checkpoints/train_logs/final_subj01_pretrained_40sess_24bs")
    parser.add_argument("--unclip_ckpt", default="checkpoints/unclip6_epoch0_step110000.ckpt")
    
    # Model config (default for 4096-dim subj01 model)
    parser.add_argument("--mindeye_hidden_dim", type=int, default=4096, help="Hidden dimension (must match model)")
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--blurry_recon", action="store_true", default=True, help="Enable blurry reconstruction branch (matches 4096-dim subj01 training)")
    
    # Processing
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test samples to process")
    parser.add_argument("--sample_indices", type=int, nargs='+', default=None, 
                        help="Specific sample indices to process")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Output
    parser.add_argument("--output_dir", default="outputs/real_reconstructions_1024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset_mean", type=float, default=0.0, help="Manually offset fMRI mean for debugging")
    parser.add_argument("--mindeye_norm_path", type=str, default="/media/hung/data1/codes/synfmri/SynBrain/src/mindeye2/norm_mean_scale_sub1.npz", help="Path to normalization stats")
    parser.add_argument("--no_norm", action="store_true", help="Skip MindEye normalization (use if input is already normalized)")
    
    # HDF5 Support
    parser.add_argument("--data_source", choices=["npy", "hdf5"], default="npy", help="Data source type")
    parser.add_argument("--hdf5_path", default="data/betas_all_subj01_fp32_renorm.hdf5", help="Path to HDF5 betas file")
    
    return parser.parse_args()


def load_mindeye2(args, num_voxels, device):
    """Load MindEye2 components with memory optimization"""
    print(f"\nLoading MindEye2 (Hidden={args.mindeye_hidden_dim}, Blurry={args.blurry_recon})...")
    
    # Clear CUDA cache first
    torch.cuda.empty_cache()
    gc.collect()
    
    # Import MindEye2 modules
    from models import BrainNetwork, PriorNetwork, BrainDiffusionPrior
    
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
    
    # Build model container
    class MindEyeModule(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return x
    
    model = MindEyeModule()
    
    # Ridge Regression
    model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)
    
    # Brain Network (backbone)
    # Important: Pass blurry_recon=False to match the trained model architecture
    model.backbone = BrainNetwork(
        h=hidden_dim, 
        in_dim=hidden_dim, 
        seq_len=1,
        clip_size=clip_emb_dim, 
        out_dim=clip_emb_dim * clip_seq_dim,
        blurry_recon=args.blurry_recon 
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
    if not os.path.exists(ckpt_path):
        # Fallback to checking inside the directory if path arg was just dir
        if os.path.isdir(os.path.join(PROJECT_ROOT, args.mindeye_ckpt)):
             ckpt_path = os.path.join(PROJECT_ROOT, args.mindeye_ckpt, "last.pth")
    
    print(f"  Loading MindEye2 checkpoint: {ckpt_path}")
    try:
        # Load CPU mapped with weights_only=False to allow deepspeed globals
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Filter state dict based on blurry_recon flag
        if not args.blurry_recon:
            filtered_state = {k: v for k, v in state_dict.items() 
                             if 'blurry' not in k.lower() and 'blur' not in k.lower() 
                             and 'blin' not in k.lower() and 'bdropout' not in k.lower()
                             and 'bnorm' not in k.lower() and 'bupsampler' not in k.lower()
                             and 'b_maps' not in k.lower()}
        else:
            filtered_state = state_dict

        # Handle Ridge Layer mismatch (e.g. using multisubject model on held-out subject)
        # Check ridge shape compatibility
        if 'ridge.linears.0.weight' in filtered_state:
             ckpt_voxels = filtered_state['ridge.linears.0.weight'].shape[1]
             if ckpt_voxels != num_voxels:
                 print(f"  [WARNING] Ridge layer shape mismatch! Checkpoint has {ckpt_voxels}, Model needs {num_voxels}.")
                 print("  Dropping ridge layer from checkpoint. Using RANDOM/UNTRAINED ridge weights.")
                 print("  Output image will likely be noise unless you fine-tune this model.")
                 filtered_state = {k: v for k, v in filtered_state.items() if 'ridge' not in k}
            
        model.load_state_dict(filtered_state, strict=False)
        del checkpoint, state_dict
        print("  MindEye2 checkpoint loaded!")
    except Exception as e:
        print(f"  Error loading MindEye2: {e}")
        return None, None, None
    
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
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"UnClip checkpoint not found at {ckpt_path}")

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
    
    print("  UnClip loaded!")
    return diffusion_engine, vector_suffix


# Import needed for unclip_recon
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



    return samples

def debug_stats(name, tensor):
    if tensor is None:
        print(f"  [DEBUG] {name}: None")
        return
    t = tensor.float()
    print(f"  [DEBUG] {name}: Shape={t.shape}, Min={t.min().item():.4f}, Max={t.max().item():.4f}, Mean={t.mean().item():.4f}, Std={t.std().item():.4f}, NaNs={torch.isnan(t).any().item()}")

def reconstruct_image(voxels, mindeye_model, diffusion_engine, vector_suffix, device, blurry_recon=False):
    """Reconstruct image from fMRI voxels using MindEye2"""
    # MindEye2 Inference
    # Ensure inputs are bf16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # Ensure correct shape [B, seq, D]
        if voxels.dim() == 1:
            voxels = voxels.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        elif voxels.dim() == 2:
            voxels = voxels.unsqueeze(1)  # [B, 1, D]
        
        voxels = voxels.to(device).to(dtype=torch.bfloat16)
        debug_stats("Input Voxels", voxels)
        
        # Ridge Regression
        voxel_ridge = mindeye_model.ridge(voxels, 0)  # [B, 1, hidden_dim]
        debug_stats("Ridge Out", voxel_ridge)
        
        # Brain Network (runs in BF16 via autocast)
        if blurry_recon:
            backbone, clip_voxels, blurry_image_enc = mindeye_model.backbone(voxel_ridge)
        else:
            # When blurry_recon is False, backbone returns only 2 values: backbone, clip_voxels
            backbone_out = mindeye_model.backbone(voxel_ridge)
            if len(backbone_out) == 3:
                backbone, clip_voxels, _ = backbone_out
            else:
                backbone, clip_voxels = backbone_out
        
        debug_stats("Backbone Out", backbone)
        
        # Diffusion Prior
        prior_out = mindeye_model.diffusion_prior.p_sample_loop(
            backbone.shape,
            text_cond=dict(text_embed=backbone),
            cond_scale=1.,
            timesteps=20
        )
        debug_stats("Prior Out Raw", prior_out)
        
        # Check/Normalize Prior Out
        prior_norm = prior_out.norm(dim=-1, keepdim=True)
        # Normalize it
        prior_out = prior_out / (prior_norm + 1e-6)
        debug_stats("Prior Out Norm", prior_out)
        
        # UnClip reconstruction
        samples = unclip_recon(
            prior_out,
            diffusion_engine,
            vector_suffix,
            num_samples=1
        )
        debug_stats("UnClip Samples", samples)
        
    return samples


def mindeye_normalize(fmri, subj, device, offset_mean=0.0):
    """
    Normalize fMRI data using pre-calculated mean and scale from MindEye2.
    """
    # Hardcoded path for now to where we found them
    norm_path = f"/media/hung/data1/codes/synfmri/SynBrain/src/mindeye2/norm_mean_scale_sub{subj}.npz"
    
    if not os.path.exists(norm_path):
        print(f"  [WARNING] Normalization file not found: {norm_path}. Using standard normalization.")
        # Fallback to standard norm if file missing
        return (fmri - fmri.mean()) / (fmri.std() + 1e-6)
        
    try:
        norm_params = np.load(norm_path)
        norm_mean = torch.tensor(norm_params['mean'], dtype=torch.float32, device=device)
        norm_scale = torch.tensor(norm_params['scale'], dtype=torch.float32, device=device)
        
        # Ensure correct shape for broadcasting
        if fmri.dim() == 3: # [B, 1, D]
            norm_mean = norm_mean.view(1, 1, -1)
            norm_scale = norm_scale.view(1, 1, -1)
        elif fmri.dim() == 2: # [B, D]
            norm_mean = norm_mean.view(1, -1)
            norm_scale = norm_scale.view(1, -1)
            
        # Scaling correction: test_fmri_avg appears to be Sum of 3 trials, not Mean
        # Its variance is ~3x the single-trial training variance.
        # Dividing by 3 brings it to the correct scale for Z-scoring.
        fmri = fmri / 3.0
        
        if offset_mean != 0.0:
            fmri = fmri + offset_mean
            
        fmri = (fmri - norm_mean) / norm_scale
        print(f"  [INFO] Applied MindEye normalization. New mean: {fmri.mean().item():.3f}, std: {fmri.std().item():.3f}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to normalize: {e}")
        
    return fmri


def create_fmri_lineplot(fmri, title="fMRI", ylim=None):
    """Create a 1D line plot visualization of fMRI data"""
    data = fmri.flatten()
    
    fig, ax = plt.subplots(figsize=(6, 2))  # Wider aspect ratio for time series/vector
    ax.plot(data, linewidth=0.5, color='black', alpha=0.8)
    # ax.set_title(title, fontsize=10) # Remove title to save space
    
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
    """Load test data: fMRI and image IDs"""
    data_dir = os.path.join(project_root, args.data_root, args.subject)
    
    # Load test fMRI
    fmri_path = os.path.join(data_dir, "test_fmri_avg.npy")
    if os.path.exists(fmri_path):
        test_fmri = np.load(fmri_path).astype(np.float32)
        print(f"Loaded test fMRI: {test_fmri.shape}")
    else:
        raise FileNotFoundError(f"Test fMRI not found: {fmri_path}")
    
    # Load image IDs
    imgid_path = os.path.join(data_dir, "test_imgIds.npy")
    if os.path.exists(imgid_path):
        test_imgids = np.load(imgid_path)
        print(f"Loaded image IDs: {test_imgids.shape}")
    else:
        test_imgids = np.arange(len(test_fmri))
        print("Using sequential image IDs")
    
    return test_fmri, test_imgids


def load_hdf5_data(args):
    """Load data directly from MindEye2 HDF5 file"""
    if not os.path.exists(args.hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {args.hdf5_path}")
        
    print(f"Loading HDF5 data from {args.hdf5_path}...")
    f = h5py.File(args.hdf5_path, 'r')
    if 'betas' not in f:
        raise KeyError(f"'betas' key not found in {args.hdf5_path}")
        
    # We will load samples on demand to save memory, so we return the dataset object
    # For compatibility with the rest of the script which expects a full array for 'test_fmri',
    # we might need to handle this carefully.
    # To keep it simple for now, we'll return the dataset handle and handle sampling in main.
    
    betas_dataset = f['betas']
    total_samples = betas_dataset.shape[0]
    print(f"  HDF5 Dataset Shape: {betas_dataset.shape}")
    
    return betas_dataset, total_samples



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
    
    # Load test data
    print("\n" + "="*60)
    print("Loading test data...")
    print("="*60)
    
    test_fmri = None
    test_imgids = None
    hdf5_dataset = None
    
    if args.data_source == "npy":
        test_fmri, test_imgids = load_test_data(args, PROJECT_ROOT)
        fmri_dim = test_fmri.shape[1]
        n_total = len(test_fmri)
    else:
        # HDF5 Mode
        hdf5_dataset, n_total = load_hdf5_data(args)
        fmri_dim = hdf5_dataset.shape[1]
        # In HDF5 mode without behavioral data, we don't have valid image IDs
        # We will use dummy IDs
        test_imgids = np.zeros(n_total, dtype=int) - 1
    
    # Determine which samples to process
    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    else:
        n_samples = min(args.num_samples, n_total)
        # For HDF5, picking random indices from 0 to 30000 might pick training data
        # But for "testing if it works", random samples are fine.
        sample_indices = np.random.choice(n_total, n_samples, replace=False)
        sample_indices.sort()

    
    print(f"\nProcessing samples: {sample_indices}")
    
    
    # Load MindEye2
    print("\n" + "="*60)
    print("Stage 1: Loading MindEye2")
    print("="*60)
    mindeye_model, clip_seq_dim, clip_emb_dim = load_mindeye2(args, fmri_dim, device)
    if mindeye_model is None:
        print("Failed to load MindEye2. Exiting.")
        return
    
    # Load UnClip
    diffusion_engine, vector_suffix = load_unclip(args, device)
    
    # Reconstruct images
    print("\n" + "="*60)
    print("Stage 2: Reconstructing images from Real fMRI")
    print("="*60)
    
    reconstructed_images = []
    
    for i, idx in enumerate(tqdm(sample_indices, desc="Reconstruction")):
        # Load sample
        if args.data_source == "npy":
            real_voxels = torch.from_numpy(test_fmri[idx:idx+1]).to(device)
        else:
            # HDF5 Load
            # Load specific index
            sample_data = hdf5_dataset[idx]
            real_voxels = torch.from_numpy(sample_data).unsqueeze(0).to(device)
            real_voxels = real_voxels.float() # Ensure float32

        # Apply normalization
        # NOTE:
        #   * For NPZ/processed NSD data ("npy" mode), we use SynBrain MindEye2
        #     stats in `mindeye_normalize` (which assumes test_fmri_avg is sum of
        #     3 trials and divides by 3).
        #   * For MindEye2 HDF5 betas ("hdf5" mode), the data in
        #     betas_all_subjXX_fp32_renorm.hdf5 are ALREADY standard-scaled per
        #     voxel during dataset creation (see MindEyeV2 `dataset_creation.ipynb`).
        #     Re-normalizing them with SynBrain stats was causing a large
        #     distribution shift and noisy reconstructions, so we skip it by
        #     default here.
        if not args.no_norm:
            subject_id = int(args.subject.replace("subj", ""))

            if args.data_source == "npy":
                # SynBrain / CortexFlow processed NPZ data
                real_voxels = mindeye_normalize(real_voxels, subject_id, device, args.offset_mean)
            else:
                # MindEye2 HDF5 betas: already normalized
                # Keep values as-is to match MindEye2 training pipeline.
                print("  [INFO] HDF5 betas detected; skipping external normalization (already renormed in MindEye2 dataset).")
        
        try:
            recon = reconstruct_image(
                real_voxels, 
                mindeye_model, 
                diffusion_engine, 
                vector_suffix, 
                device,
                blurry_recon=args.blurry_recon
            )
            reconstructed_images.append(recon.cpu())
        except Exception as e:
            print(f"  Error reconstructing sample {idx}: {e}")
            reconstructed_images.append(torch.zeros(1, 3, 768, 768))
        
        # Clear cache after each sample
        torch.cuda.empty_cache()
    
    # Create comparison visualizations
    print("\n" + "="*60)
    print("Stage 3: Creating comparison visualizations")
    print("="*60)
    
    # Calculate robust limits for fMRI plotting
    if args.data_source == "npy":
        all_real = test_fmri[sample_indices]
    else:
        # For HDF5, just use the first sample to estimate range to avoid loading too much
        all_real = hdf5_dataset[sample_indices[0]]

    vmin = np.percentile(all_real, 1)
    vmax = np.percentile(all_real, 99)
    range_span = vmax - vmin
    ylim = (vmin - 0.1 * range_span, vmax + 0.1 * range_span)
    
    print(f"Plotting fMRI with y-limits: {ylim}")
    
    for i, idx in enumerate(sample_indices):
        print(f"Creating visualization for sample {idx}...")
        
        # Load data for plotting
        if args.data_source == "npy":
            fmri_sample = test_fmri[idx]
            img_id = test_imgids[idx]
        else:
            fmri_sample = hdf5_dataset[idx]
            img_id = -1 # No valid image ID for random HDF5 samples yet
            
        # Load original image
        if img_id != -1:
            original_img = load_nsd_image(img_id, args.nsd_path, PROJECT_ROOT)
        else:
             # Create placeholder for unknown image
            placeholder = np.zeros((512, 512, 3), dtype=np.uint8) 
            placeholder[:] = 20 # Dark gray background
            # Add some white noise so it's not just black
            noise = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            placeholder = (placeholder * 0.8 + noise * 0.2).astype(np.uint8)
            original_img = Image.fromarray(placeholder)
            
        original_img = original_img.resize((512, 512))
        
        # Create fMRI line plot
        real_fmri_img = create_fmri_lineplot(fmri_sample, "Real fMRI", ylim)
        
        # Get reconstructed image
        if i < len(reconstructed_images):
            recon_tensor = reconstructed_images[i][0]
            if recon_tensor.dtype == torch.bfloat16:
                recon_tensor = recon_tensor.float()
            
            # Save raw tensor for debugging
            torch.save(recon_tensor, os.path.join(args.output_dir, f"raw_tensor_{idx}.pt"))
            
            recon_img = transforms.ToPILImage()(recon_tensor)
            
            # Debug PIL stats
            extrema = recon_img.getextrema()
            print(f"  [DEBUG] PIL Extrema sample {idx}: {extrema}")
            stat = np.array(recon_img)
            print(f"  [DEBUG] PIL Mean sample {idx}: {stat.mean():.2f}")
            
            # Save raw image
            recon_img.save(os.path.join(args.output_dir, f"raw_recon_{idx}.png"))
            
            recon_img = recon_img.resize((512, 512))
        else:
            recon_img = Image.new('RGB', (512, 512), color='gray')
        
        # Create comparison figure (Original | line plot | Recon)
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.1)
        
        # 1. Original Image
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(original_img)
        ax0.set_title('Original Image', fontsize=14)
        ax0.axis('off')
        
        # 2. Real fMRI Line Plot
        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(real_fmri_img)
        ax1.set_title(f'Real fMRI (Sample {idx})', fontsize=14)
        ax1.axis('off')
        
        # 3. Reconstructed Image
        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(recon_img)
        ax2.set_title(f'MindEye2 (1024-dim)\nReconstruction', fontsize=14)
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(args.output_dir, f"real_recon_sample_{idx:04d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path}")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print(f"Output directory: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
