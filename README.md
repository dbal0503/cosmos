# 🚀 COSMOS: Compressed and Smooth Latent Space for Text Diffusion Modeling

This repository contains the official implementation for our paper **[Compressed and Smooth Latent Space for Text Diffusion Modeling](https://arxiv.org/abs/2506.21170)**, which was accepted as a poster at NeurIPS 2025.

While autoregressive models dominate text generation, their sequential nature leads to slow decoding and challenges in maintaining global coherence. Diffusion models offer a parallelizable alternative, but their application to text is hindered by the high dimensionality of token-level representations.

We introduce **COSMOS**, a novel approach that operates entirely in a compressed, smooth latent space. This space is learned using an autoencoder trained for both token-level reconstruction and alignment with a pretrained language encoder, providing robust semantic grounding. Our method allows for an **8x compression** of text representations while maintaining high quality, achieving comparable or superior results to strong baselines.

## 📋 Table of Contents

- [🔧 Environment Setup](#-environment-setup)
- [📊 Dataset Preparation](#-dataset-preparation)
- [🎯 Training](#-training)
- [🧬 Generation](#-generation)
- [📝 How to Cite](#-how-to-cite)

---

## 🔧 Environment Setup

To get started, set up a virtual environment and install the required dependencies using `uv`:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Set the project root directory
export PROJECT_ROOT=$(pwd)

# Install dependencies
uv sync
```

You will also need to authorize with Weights & Biases for experiment tracking:

```bash
wandb login
```

---

## 📊 Dataset Preparation

For convenience, all datasets used in the paper have been pre-processed and are available on the Hugging Face Hub. **We recommend using these pre-processed datasets.** You can find all of them here: [bayes-group-diffusion/datasets](https://huggingface.co/bayes-group-diffusion).

The training scripts will automatically download and save them in the `data` directory. You just need to ensure the `dataset` in your configuration file (`conf/config.yaml`) points to the correct dataset. For example, for `rocstories`:

```yaml
# in conf/config.yaml
- dataset: "rocstories"
```

```bash
# or uv run python -m utils.load_to_hub --config_path ../conf/ --load_from_hub
python -m utils.load_to_hub --config_path ../conf/ --load_from_hub
```

To use other datasets, update the configuration file accordingly:

```yaml
# in conf/config.yaml
- dataset: "wikipedia" # or "openwebtext-128", "openwebtext-512"
```

An example of the data preprocessing script is available at `utils/owt_preparation.py`. The Wikipedia and OpenWebText datasets were prepared using a similar process, mainly differing in the text chunk length.

---

## Pretrained Models

We provide pretrained model checkpoints on AWS S3. To download them, you will need to have the AWS CLI installed and configured.

### Prerequisites

First, install the necessary packages to interact with AWS:

```bash
pip install boto3 awscli
```

Next, configure your AWS credentials. If you haven't done this before, run the following command and follow the prompts:

```bash
aws configure
```

### Available Checkpoints

**1. Autoencoder**

To download the autoencoder, create the destination directory and run the copy command:
```bash
mkdir -p ./checkpoints/autoencoder-num_latents=16-wikipedia-final-128/

aws s3 cp s3://cosmos-latent-diffusion/checkpoints/autoencoder-num_latents=16-wikipedia-final-128/100000.pth ./checkpoints/autoencoder-num_latents=16-wikipedia-final-128/100000.pth --region eu-north-1
```

Available checkpoints in S3 `cosmos-latent-diffusion/checkpoints`:
1. `autoencoder-num_latents=16-wikipedia-final-128/100000.pth`
2. `autoencoder-num_latents=32-wikipedia-final-128/100000.pth`
3. `autoencoder-num_latents=64-wikipedia-final-128/100000.pth`
4. `autoencoder-num_latents=128-wikipedia-final-128/100000.pth`
5. `autoencoder-num_latents=512-openwebtext-512-final-512/200000.pth`

The name of the checkpoint means:
- `rocstories`: dataset name
- `num_latents=16`: number of latents
- `wikipedia`: dataset name
- `final`: final checkpoint
- `128`: max sequence length

**2. Diffusion Model**

To download the diffusion model, create the destination directory and run the copy command:
```bash
mkdir -p ./checkpoints/diffusion-rocstories-16-d=5-final/

aws s3 cp s3://cosmos-latent-diffusion/checkpoints/diffusion-rocstories-16-d=5-final/180000.pth ./checkpoints/diffusion-rocstories-16-d=5-final/180000.pth --region eu-north-1
```

Available checkpoints in S3 `cosmos-latent-diffusion/checkpoints`:
1. `diffusion-rocstories-16-d=5-final/180000.pth`
2. `diffusion-rocstories-32-d=5-final/200000.pth`
3. `diffusion-rocstories-64-d=7-final/200000.pth`
4. `diffusion-openwebtext-512-512-d=5-final-512/500000.pth`

The name of the checkpoint means:
- `rocstories`: dataset name
- `num_latents=16`: number of latents
- `d=5`: scheduler parameter
- `final`: final checkpoint

---

## 🎯 Training


The training process consists of two main stages: training the autoencoder and training the diffusion model.

### 1. Autoencoder Training

Train the autoencoder to learn a compressed latent representation of the text:

```bash
HYDRA_FULL_ERROR=1 \
uv run \
torchrun --nproc_per_node=4 --master_port=12346 train_encoder.py \
dataset=wikipedia \
encoder.latent.num_latents=16 \
decoder.latent.num_latents=16 \
encoder.augmentation.masking.weight=0.5 \
encoder.augmentation.masking.encodings_mlm_probability=0.3 \
encoder.augmentation.gaussian_noise.weight=0.5 \
encoder.augmentation.gaussian_noise.delta=0.7 \
encoder.augmentation.latent_masking.probability=0.4 \
autoencoder.latent.dim=768 \
autoencoder.latent.num_latents=16 \
training.training_iters=100000 \
training="autoencoder" \
suffix="final"
```

### 2. Diffusion Model Training

Once the autoencoder is trained, use its weights to train the diffusion model on the latent space:

```bash
CUDA_LAUNCH_BLOCKING=1 \
HYDRA_FULL_ERROR=1 \
uv run \
torchrun --nproc_per_node=4 --master_port=12345 \
train_diffusion.py \
dataset=rocstories \
diffusion.dynamic.N=200 \
diffusion.dynamic.d=5 \
diffusion.training.batch_size=512 \
encoder.latent.num_latents=16 \
encoder.embedding.max_position_embeddings=128 \
decoder.latent.num_latents=16 \
decoder.embedding.max_position_embeddings=128 \
autoencoder.model.load_checkpoint='"autoencoder-num_latents=16-wikipedia-final-128/100000.pth"' \
diffusion.generation.num_gen_texts=2000 \
training=diffusion \
suffix="final"
```

---

## ✍️ Generation

After training the diffusion model, you can generate new text samples:

```bash
CUDA_LAUNCH_BLOCKING=1 \
HYDRA_FULL_ERROR=1 \
uv run \
torchrun --nproc_per_node=4 --master_port=12345 \
generate.py \
dataset=rocstories \
diffusion.dynamic.N=200 \
diffusion.dynamic.d=5 \
diffusion.training.batch_size=512 \
encoder.latent.num_latents=16 \
encoder.embedding.max_position_embeddings=128 \
decoder.latent.num_latents=16 \
decoder.embedding.max_position_embeddings=128 \
autoencoder.model.load_checkpoint='"autoencoder-num_latents=16-wikipedia-final-128/100000.pth"' \
diffusion.model.load_checkpoint='"diffusion-rocstories-16-d=5-final/180000.pth"' \
diffusion.generation.num_gen_texts=2000 \
training=""
```

---

## 📁 Project Structure

```
cosmos/
├── 📁 conf/                # Hydra configuration files
├── 📁 estimation/          # Metrics and quality assessment code
├── 📁 utils/               # Data preparation utilities and logging utilities
├── 📁 architecture/        # Model architectures
├── 📁 diffusion_utils/     # Diffusion dynamic, scheduler, and solver
├── 📁 diffusion_trainer.py # Diffusion trainer main class
├── 📁 encoder_trainer.py   # Encoder trainer main class
├── 🐍 train_encoder.py     # Script for training the autoencoder
├── 🐍 train_diffusion.py   # Script for training the diffusion model
└── 🐍 generate.py          # Script for text generation
```

---

## 📝 How to Cite

If you use this work, please cite our paper:

```bibtex

```

---

## 🤝 Collaboration

If you are interested in collaborating, please reach out to us at [meshchaninov.viacheslav@gmail.com](mailto:meshchaninov.viacheslav@gmail.com) or [vmeshchani@constructor.university](mailto:vmeshchani@constructor.university).
