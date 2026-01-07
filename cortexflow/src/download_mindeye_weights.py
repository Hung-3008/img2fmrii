
import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files

# Configuration
REPO_ID = "pscotti/mindeyev2"
REPO_TYPE = "dataset"

# Base paths (assuming this script is run from .../synfmri/ or script calculates path relative to this file)
# We want to support running from anywhere, so we rely on finding the project root relative to this file.
# File location: cortexflow/src/download_mindeye_weights.py
# Project root: ../../ (relative to this file)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "MindEyeV2", "data")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MINDEYE_ROOT = os.path.join(PROJECT_ROOT, "MindEyeV2")

# Define resources
RESOURCES = {
    "autoenc": {
        "type": "file",
        "filename": "sd_image_var_autoenc.pth",
        "target_dir": DATA_DIR,
        "desc": "SD Image Variation Autoencoder"
    },
    "betas": {
        "type": "file",
        "filename": "betas_all_subj01_fp32_renorm.hdf5",
        "target_dir": DATA_DIR,
        "desc": "Normalized fMRI Betas for Subject 01"
    },
    "coco": {
        "type": "file",
        "filename": "coco_images_224_float16.hdf5",
        "target_dir": DATA_DIR,
        "desc": "COCO Images (224x224, float16)"
    },
    "converter": {
        "type": "file",
        "filename": "bigG_to_L_epoch8.pth",
        "target_dir": DATA_DIR,
        "desc": "BigG to L Converter Model"
    },
    "multisubject_model": {
        "type": "folder",
        "pattern": "train_logs/multisubject_subj01_1024hid_nolow_300ep/*",
        "target_dir": CHECKPOINTS_DIR,
        "desc": "Multisubject Model Checkpoint (Subj01, 1024hid, NoLow, 300ep)"
    },
    "wds_test": {
        "type": "custom_wds_test",
        "target_dir": DATA_DIR,
        "desc": "WebDataset Test Files for Subject 01"
    },
    "weights": {
        "type": "file",
        "filename": "train_logs/final_subj01_pretrained_40sess_24bs/last.pth",
        "target_dir": MINDEYE_ROOT, # Keeping legacy location
        "desc": "Main Pretrained Weights (Subj01, 40sess, 24bs)"
    }
}

def download_file(resource_key, config, dry_run=False):
    target_path = os.path.abspath(config["target_dir"])
    print(f"[{resource_key}] Downloading {config['filename']} to {target_path}...")
    
    if dry_run:
        print(f"  [Dry Run] Would call hf_hub_download(filename={config['filename']}, local_dir={target_path})")
        return

    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=config["filename"],
            repo_type=REPO_TYPE,
            local_dir=target_path
        )
        print(f"  Successfully downloaded to {os.path.join(target_path, config['filename'])}")
    except Exception as e:
        print(f"  Error downloading {resource_key}: {e}")

def download_folder(resource_key, config, dry_run=False):
    target_path = os.path.abspath(config["target_dir"])
    print(f"[{resource_key}] Downloading folder matching '{config['pattern']}' to {target_path}...")

    if dry_run:
        print(f"  [Dry Run] Would call snapshot_download(allow_patterns={config['pattern']}, local_dir={target_path})")
        return

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=target_path,
            allow_patterns=config["pattern"],
            resume_download=True
        )
        print(f"  Successfully downloaded folder contents to {target_path}")
    except Exception as e:
        print(f"  Error downloading {resource_key}: {e}")

def download_wds_test(resource_key, config, dry_run=False):
    target_dir_suffix = "wds/subj01/new_test"
    target_path = os.path.abspath(config["target_dir"])
    print(f"[{resource_key}] Downloading WebDataset files from {target_dir_suffix} to {target_path}...")

    if dry_run:
        print(f"  [Dry Run] Would list files in {target_dir_suffix} and download .tar files to {target_path}")
        return

    try:
        all_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
        test_files = [f for f in all_files if f.startswith(target_dir_suffix) and f.endswith(".tar")]

        if not test_files:
            print(f"  No .tar files found in {target_dir_suffix}")
            return

        print(f"  Found {len(test_files)} files.")
        for file_path in test_files:
            print(f"  Downloading {file_path}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                repo_type=REPO_TYPE,
                local_dir=target_path
            )
        print(f"  Successfully downloaded all wds_test files to {target_path}")

    except Exception as e:
         print(f"  Error downloading {resource_key}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download MindEyeV2 weights and data.")
    parser.add_argument("resources", nargs="*", help="List of resources to download (or 'all'). Available: " + ", ".join(RESOURCES.keys()))
    parser.add_argument("--list", action="store_true", help="List all available resources.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate the download process without actually downloading.")

    args = parser.parse_args()

    if args.list:
        print("Available resources:")
        for key, val in RESOURCES.items():
            print(f"  - {key}: {val['desc']}")
        return

    keys_to_download = []
    if not args.resources:
        print("No resources specified. Use --list to see available options, or pass 'all' to download everything.")
        return
    
    if "all" in args.resources:
        keys_to_download = list(RESOURCES.keys())
    else:
        for key in args.resources:
            if key in RESOURCES:
                keys_to_download.append(key)
            else:
                print(f"Warning: Resource '{key}' not found. Skipping.")

    print(f"Project Root detected as: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Checkpoints Directory: {CHECKPOINTS_DIR}")
    print("-" * 50)

    for key in keys_to_download:
        config = RESOURCES[key]
        if config["type"] == "file":
            download_file(key, config, args.dry_run)
        elif config["type"] == "folder":
            download_folder(key, config, args.dry_run)
        elif config["type"] == "custom_wds_test":
            download_wds_test(key, config, args.dry_run)
        print("-" * 50)

if __name__ == "__main__":
    main()
