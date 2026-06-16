#!/bin/bash

# Set the checkpoints directory
CheckpointsDir="models"

# Create necessary directories
mkdir -p models/zen-dub models/zen-dubV15 models/syncnet models/dwpose models/face-alignment models/sd-vae models/whisper

# Install required packages
pip install -U "huggingface_hub[cli]"

# Set HuggingFace mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com

# Download Zen-Dub V1.0 weights
huggingface-cli download hanzoai/Zen-Dub \
  --local-dir $CheckpointsDir \
  --include "zen-dub/zen-dub.json" "zen-dub/pytorch_model.bin"

# Download Zen-Dub V1.5 weights (unet.pth)
huggingface-cli download hanzoai/Zen-Dub \
  --local-dir $CheckpointsDir \
  --include "zen-dubV15/zen-dub.json" "zen-dubV15/unet.pth"

# Download SD VAE weights
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Download Whisper weights
huggingface-cli download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Download DWPose weights
huggingface-cli download yzd-v/DWPose \
  --local-dir $CheckpointsDir/dwpose \
  --include "dw-ll_ucoco_384.pth"

# Download SyncNet weights
huggingface-cli download ByteDance/LatentSync \
  --local-dir $CheckpointsDir/syncnet \
  --include "latentsync_syncnet.pt"

# Download 2D-FAN-4 facial-landmark weights (used to build the face blend mask)
curl -L https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip \
  -o $CheckpointsDir/face-alignment/2DFAN4-cd938726ad.pth

echo "All weights have been downloaded successfully!"
