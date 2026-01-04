import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainAutoencoder(nn.Module):
    """Simple MLP-based brain autoencoder.

    Encodes flat fMRI vector [B, D] -> latent [B, latent_dim],
    then decodes latent back to [B, D].
    """

    def __init__(self,
                 input_dim: int = 15724,
                 hidden_dims=None,
                 latent_dim: int = 1024,
                 dropout: float = 0.0):
        super().__init__()

        if hidden_dims is None:
            # One moderate hidden layer by default
            hidden_dims = [2048]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder MLP: input_dim -> hidden_dims -> latent_dim
        enc_layers = []
        last_dim = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(last_dim, h))
            enc_layers.append(nn.LayerNorm(h))
            enc_layers.append(nn.GELU())
            if dropout > 0.0:
                enc_layers.append(nn.Dropout(dropout))
            last_dim = h
        enc_layers.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder MLP: latent_dim -> reversed hidden_dims -> input_dim
        dec_layers = []
        last_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(last_dim, h))
            dec_layers.append(nn.LayerNorm(h))
            dec_layers.append(nn.GELU())
            last_dim = h
        dec_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Tensor of shape [B, D] or [B, ..., D].

        Returns:
            recon: [B, D]
            z:     [B, latent_dim]
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class BrainVAE(nn.Module):
    """VAE-style brain autoencoder with CLIP alignment.

    Encodes flat fMRI vector [B, D] -> latent [B, latent_dim] (Gaussian),
    decodes back to [B, D], and optionally aligns latent with CLIP embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        clip_dim: int,
        hidden_dims=None,
        latent_dim: int = 512,
        dropout: float = 0.1,
        kl_weight: float = 1.0,
        clip_weight: float = 0.0,
        temp: float = 0.125,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [2048]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.clip_weight = clip_weight
        self.temp = temp

        # Encoder MLP backbone (without final latent layer)
        enc_layers = []
        last_dim = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(last_dim, h))
            enc_layers.append(nn.LayerNorm(h))
            enc_layers.append(nn.GELU())
            if dropout > 0.0:
                enc_layers.append(nn.Dropout(dropout))
            last_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # Latent mean and log-variance
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)

        # Decoder MLP: latent_dim -> reversed hidden_dims -> input_dim
        dec_layers = []
        last_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(last_dim, h))
            dec_layers.append(nn.LayerNorm(h))
            dec_layers.append(nn.GELU())
            last_dim = h
        dec_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Project CLIP embeddings into latent space for alignment loss
        self.clip_proj = nn.Linear(clip_dim, latent_dim) if clip_dim is not None else None

    def encode(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar, sample_posterior: bool = True):
        if not sample_posterior:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def recon_loss(self, recon, x):
        return F.mse_loss(recon, x)

    def kl_loss(self, mu, logvar):
        # Standard VAE KL divergence to N(0, I)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum() / mu.size(0)

    def soft_clip_loss(self, preds, targs):
        # Adapted from SynBrain BrainVAE: contrastive alignment between
        # brain latents (preds) and CLIP embeddings (targs).
        preds = F.normalize(preds.view(preds.size(0), -1), dim=-1)
        targs = F.normalize(targs.view(targs.size(0), -1), dim=-1)

        temp = self.temp
        clip_clip = (targs @ targs.T) / temp
        brain_clip = (preds @ targs.T) / temp

        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        return (loss1 + loss2) / 2.0

    def forward(self, x, clip=None, sample_posterior: bool = True):
        """Forward pass.

        Args:
            x:    fMRI tensor [B, D] or [B, ..., D]
            clip: CLIP embeddings [B, C] (optional)
            sample_posterior: if False, use mean instead of sampling

        Returns:
            recon:       [B, D]
            z:           [B, latent_dim]
            recon_loss:  scalar
            kl_loss:     scalar
            clip_loss:   scalar
            total_loss:  scalar
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sample_posterior=sample_posterior)
        recon = self.decode(z)

        rec_loss = self.recon_loss(recon, x)
        kl = self.kl_loss(mu, logvar)

        if self.clip_weight > 0.0 and clip is not None and self.clip_proj is not None:
            if clip.dim() > 2:
                clip = clip.view(clip.size(0), -1)
            clip_latent = self.clip_proj(clip)
            clip_loss = self.soft_clip_loss(z, clip_latent)
        else:
            clip_loss = x.new_tensor(0.0)

        total_loss = rec_loss + self.kl_weight * kl + self.clip_weight * clip_loss
        return recon, z, rec_loss, kl, clip_loss, total_loss

