# Tests for Brain Autoencoder

This folder contains small utility scripts for checking the BrainAutoencoder in `cortexflow`.

## `brain_autoencoder_eval.py`

Evaluates a trained `BrainAutoencoder` on the test split and compares it against a simple zero baseline (always predicting 0).

Example usage from the repository root:

```bash
python test/brain_autoencoder_eval.py \
  --data_root data/processed \
  --subject subj01 \
  --ckpt checkpoints/brain_vae_v1/best_ae.pth
```

The script prints:
- Test data statistics (mean, std, min, max)
- Zero-baseline MSE
- Autoencoder test MSE and average Pearson correlation
- Reconstruction statistics (mean/std/min/max) on the first batch

Use these numbers to verify that the autoencoder is learning better than the zero baseline and producing non-trivial reconstructions.
