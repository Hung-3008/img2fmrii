# CortexFlow-SP001: A Simplified Flow-Based NSD Encoding Model

*Version: SP001 (simple prototype)*  
*Target: single-ROI, single-subject NSD encoding with conditional flow on fMRI latents*

---

## 1. Motivation and Scope

CortexFlow-SP001 is a **minimal, implementable prototype** of the CortexFlow idea for NSD.  
Instead of modeling full-brain voxel patterns with a hierarchical, ROI-structured flow, SP001 focuses on:

- A **single subject** (e.g., NSD subj01).
- A **single ROI or ROI group** (e.g., ventral visual cortex or early visual cortex) with a few thousand voxels.
- A **low-dimensional fMRI latent space**, learned by an autoencoder.
- A **conditional normalizing flow** in that latent space, conditioned on frozen image embeddings (e.g., CLIP).

The goal is to answer a concrete question:

> Can a flow-based generative model over **fMRI latents** (rather than voxels) improve NSD encoding vs. deterministic baselines, while remaining simple enough to implement and train?

This prototype deliberately **omits** several complexities of the full CortexFlow:

- No explicit multi-ROI hierarchy (V1→V2→V4→IT) in the architecture.
- No explicit cortical surface topology or receptive-field constraints.
- No multi-subject subject-embedding mechanism (SP001 is single-subject; multi-subject is future work).
- No heavy RSA/Jacobian regularization; we begin with a clean maximum-likelihood objective plus simple reconstruction/correlation metrics.

---

## 2. Data and Setting

### 2.1 Dataset

- **Dataset:** Natural Scenes Dataset (NSD).  
- **Subject:** One subject (e.g., `subj01` in `nsd/` data tree).
- **Images:** Natural scenes from MS-COCO as in NSD; we use the standard NSD split (e.g., training on ~10k images, validation/test on held-out images and/or repetitions).

### 2.2 fMRI Representation

- Start from **preprocessed single-trial responses** (e.g., GLMsingle betas) for the chosen subject.
- Restrict to a **single ROI/ROI group**: e.g., "ventral stream" or "early visual" voxels.
- Flatten voxel responses for that ROI into a vector \(x \in \mathbb{R}^D\), where \(D\) is the number of voxels (e.g., \(D \approx 5k\)–10k).

### 2.3 Image Representation

- Use a **frozen vision encoder**, such as CLIP-ViT-B/32 or CLIP-RN50.
- For each NSD image, compute an image embedding \(z_{\text{img}} \in \mathbb{R}^{d_{\text{img}}}\).
- These embeddings are **precomputed** once and saved (e.g., in `SynBrain/data/` as `.npy` or `.pt`).

---

## 3. Model Overview

CortexFlow-SP001 factorizes the problem into two parts:

1. **fMRI autoencoder** \(E_{\theta}, D_{\phi}\): maps voxel patterns to a low-dimensional latent and back.
   - Encoder: \(z_{\text{brain}} = E_{\theta}(x) \in \mathbb{R}^{d_{\text{brain}}}\).
   - Decoder: \(\hat{x} = D_{\phi}(z_{\text{brain}}) \in \mathbb{R}^D\).
   - Typically \(d_{\text{brain}} \ll D\) (e.g., 64–256).

2. **Conditional flow** \(F_{\psi}\): a normalizing flow over the brain latent space, conditioned on image embeddings.
   - Base distribution: \(u \sim \mathcal{N}(0, I_{d_{\text{brain}}})\).
   - Flow: \(z_{\text{brain}} = F_{\psi}(u; z_{\text{img}})\), invertible in \(u \leftrightarrow z_{\text{brain}}\).
   - At inference: given an image embedding \(z_{\text{img}}\), we can sample one or multiple \(z_{\text{brain}}\), then decode to voxels via \(D_{\phi}\).

We treat the fMRI autoencoder as defining a **brain latent space** that is (mostly) sufficient for reconstructing voxel patterns. The flow then **learns the conditional distribution** \(p(z_{\text{brain}} \mid z_{\text{img}})\) directly in this space.

---

## 4. Architecture Details

### 4.1 fMRI Autoencoder

**Encoder \(E_{\theta}\):**

- Input: voxel vector \(x \in \mathbb{R}^D\) (standardized per-voxel across training data).
- Architecture: simple MLP or shallow network, e.g.:
  - `Linear(D, h1)` → `ReLU` → `Linear(h1, h2)` → `ReLU` → `Linear(h2, d_brain)`.
- Optionally include LayerNorm or BatchNorm to stabilize training.

**Decoder \(D_{\phi}\):**

- Input: \(z_{\text{brain}} \in \mathbb{R}^{d_{\text{brain}}}\).
- Architecture: symmetric MLP, e.g.:
  - `Linear(d_brain, h2)` → `ReLU` → `Linear(h2, h1)` → `ReLU` → `Linear(h1, D)`.
- Output: \(\hat{x}\) as predicted voxel pattern.

**Autoencoder training objective:**

We first train \(E_{\theta}, D_{\phi}\) on real fMRI data only:

- Reconstruction loss (voxel-wise MSE):
  $$
  \mathcal{L}_{\text{AE}} 
  = \mathbb{E}_{x \sim \text{NSD}} \Big[\, \lVert x - D_{\phi}(E_{\theta}(x)) \rVert_2^2 \,\Big].
  $$

We can add an L2 regularizer on weights and/or a small KL-like penalty to keep \(z_{\text{brain}}\) roughly centered (but SP001 does not require a full VAE).

After this stage, we **freeze** \(E_{\theta}, D_{\phi}\) when training the flow (at least in the simplest version). Later variants could fine-tune jointly.

### 4.2 Conditional Normalizing Flow in Latent Space

We model the distribution of fMRI latents given the image embedding:

$$
\begin{aligned}
&u \sim \mathcal{N}(0, I_{d_{\text{brain}}}),\\
&z_{\text{brain}} = F_{\psi}(u; z_{\text{img}}),\\
&x = D_{\phi}(z_{\text{brain}}).
\end{aligned}
$$

**Flow choice:**

- Use a stack of **RealNVP-style affine coupling layers** or similar (e.g., Glow-like) operating on \(z_{\text{brain}} \in \mathbb{R}^{d_{\text{brain}}}\).
- For each coupling layer, we split the latent dimension into two parts (e.g., first half and second half) and transform one part conditioned on the other plus \(z_{\text{img}}\).

**Conditioning on image embeddings:**

- For each coupling layer, use a small MLP to map \(z_{\text{img}}\) to scale/shift parameters, or incorporate it via FiLM:
  - `cond = MLP(z_img)` → modulate the internal hidden units of the scale/shift networks.
- This yields **image-conditioned** transformations: the same base noise \(u\) produces different \(z_{\text{brain}}\) for different \(z_{\text{img}}\).

**Number of layers/dimensions:**

- Latent dimension: \(d_{\text{brain}} \approx 64\)–128.
- Flow depth: e.g., 8–16 coupling layers, potentially grouped into blocks with permutations in between.

---

## 5. Training Objectives

### 5.1 Stage 1: Autoencoder Pretraining

Train \(E_{\theta}, D_{\phi}\) on fMRI only, as described above, minimizing \(\mathcal{L}_{\text{AE}}\) on the chosen subject and ROI.

- Optimize until voxel-wise reconstruction correlation (e.g., Pearson across trials) is satisfactory.
- Save encoder/decoder checkpoints and latent codes for all training trials (optional for speed).

### 5.2 Stage 2: Conditional Flow Training

With the autoencoder fixed, we now train \(F_{\psi}\) to model \(p(z_{\text{brain}} \mid z_{\text{img}})\):

1. For each training trial:
   - Take voxel vector \(x\), compute \(z_{\text{brain}} = E_{\theta}(x)\).
   - Take the corresponding image, obtain frozen embedding \(z_{\text{img}}\).
2. Use the change-of-variables formula for flows to compute the conditional log-likelihood:

   $$
   \log p(z_{\text{brain}} \mid z_{\text{img}})
   = \log p(u) + \log \left| \det \left(\frac{\partial u}{\partial z_{\text{brain}}} \right) \right|
   $$

   where \(u = F_{\psi}^{-1}(z_{\text{brain}}; z_{\text{img}})\) and \(p(u) = \mathcal{N}(0, I)\).

3. Minimize the negative log-likelihood:

   $$
   \mathcal{L}_{\text{flow}} 
   = -\mathbb{E}_{(z_{\text{brain}}, z_{\text{img}})} 
   \big[\log p(z_{\text{brain}} \mid z_{\text{img}})\big].
   $$

In SP001, we do **not** require extra RSA or CLIP-style losses. The flow is trained purely to match the empirical latent distribution conditioned on images.

### 5.3 Optional Joint Fine-Tuning (Later Variant)

Once a stable flow is obtained, a later variant (beyond SP001) could allow **joint fine-tuning** of \(E_{\theta}, D_{\phi}, F_{\psi}\) with a combined loss:

- Voxel reconstruction loss \(\mathcal{L}_{\text{AE}}\).
- Flow negative log-likelihood \(\mathcal{L}_{\text{flow}}\).
- Optional semantic alignment loss (e.g., CLIP-like alignment between decoded fMRI and image embeddings).

SP001, however, **stops before** this joint fine-tuning to keep the first prototype manageable.

---

## 6. Evaluation Plan

We evaluate CortexFlow-SP001 along three axes:

### 6.1 Voxel-Level Encoding Accuracy

- Given a test image, sample **one or more** synthetic fMRI patterns:
  - Sample \(u \sim \mathcal{N}(0, I)\), compute \(z_{\text{brain}} = F_{\psi}(u; z_{\text{img}})\), then \(\hat{x} = D_{\phi}(z_{\text{brain}})\).
- Compare \(\hat{x}\) to held-out real fMRI \(x_{\text{test}}\):
  - Voxel-wise Pearson correlation.
  - Voxel-wise coefficient of determination (\(R^2\)).
- Baselines:
  - Linear encoding: ridge regression from \(z_{\text{img}}\) to voxels.
  - Simple MLP encoder: MLP from \(z_{\text{img}}\) to voxel space.
- Metrics: improvement in correlation / \(R^2\) against baselines.

### 6.2 Latent-Space Likelihood and Calibration

- Report average conditional log-likelihood \(\log p(z_{\text{brain}} \mid z_{\text{img}})\) on held-out trials.
- Check whether the inferred base noise \(u = F_{\psi}^{-1}(z_{\text{brain}}; z_{\text{img}})\) is approximately Gaussian (e.g., via marginal histograms and simple normality tests).

### 6.3 Semantic-Level Retrieval

- Use synthetic fMRI to perform **image identification**:
  - For each test image, generate \(\hat{x}\) and pass it through the existing decoding pipeline (e.g., CLIP readout or other SynBrain decoders) to get a synthetic embedding.
  - Rank all candidate images by similarity, report top-k retrieval accuracy (e.g., top-1, top-5).
- Compare against using real fMRI and against deterministic predictions (ridge/MLP) from \(z_{\text{img}}\).

---

## 7. Relation to Full CortexFlow and Next Steps

CortexFlow-SP001 is explicitly a **stepping stone** toward the full CortexFlow vision:

- **From latent to voxel-level flow:** SP001 learns the flow in low-dimensional latent space. A next step is to increase latent dimension and/or introduce weak ROI structure in the latent (grouped dimensions per ROI).
- **From single subject to multi-subject:** After validating SP001 on one subject, we can:
  - Share the autoencoder across subjects.
  - Introduce a small subject embedding or low-rank affine transform in latent space.
  - Extend the conditional flow to take both \(z_{\text{img}}\) and subject code as inputs.
- **From flat latent to hierarchical cortex:** Once the simple latent flow works, we can:
  - Partition the latent dimensions into groups corresponding to V1, V2, V4, IT (based on voxel selection in the autoencoder).
  - Replace the single flow with a hierarchical composition of regional flows.

The success criteria for SP001 are modest but important:

- The model achieves **competitive or better** voxel-wise encoding performance vs. deterministic baselines.
- The latent-space flow is **well-calibrated** (approximate Gaussian base, meaningful likelihood).
- Sampling multiple \(z_{\text{brain}}\) per image improves performance when averaging predictions, indicating that the model captures **structured variability** rather than just noise.

If these criteria are met, they justify the additional complexity of moving toward the full CortexFlow design.

---

## 8. Implementation Notes (SynBrain Repository)

Within the existing `SynBrain` codebase, a natural implementation plan for SP001 is:

1. **Data preprocessing scripts** (e.g., `SynBrain/data/`):
   - Extract NSD single-trial betas for one subject and ROI.
   - Standardize and save voxel vectors and corresponding image indices.
   - Precompute and cache CLIP embeddings for the relevant images.

2. **Model code** (e.g., in `SynBrain/src/`):
   - Implement an fMRI autoencoder module (`BrainAutoencoder`):
     - Configurable dimensions `D`, `d_brain`, and hidden sizes.
   - Implement a conditional flow module (`LatentFlow`):
     - RealNVP-style coupling layers on \(\mathbb{R}^{d_{\text{brain}}}\).
     - Conditioning on `z_img` via small MLP/FiLM.

3. **Training scripts / notebooks**:
   - Stage 1: autoencoder training notebook/script.
   - Stage 2: conditional flow training notebook/script.
   - Evaluation notebook: voxel-wise metrics and semantic retrieval comparisons.

This SP001 proposal is intentionally narrow in scope but technically complete: it specifies the data subset, model structure, training objectives, and evaluation criteria needed to run a first CortexFlow-style experiment on NSD.
