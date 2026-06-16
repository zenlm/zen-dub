# Third-Party Licenses — zen-dub

zen-dub (native-Rust MuseTalk lip-sync video dub) is licensed under **Apache-2.0**
(see `LICENSE`). It builds on the following open-source components:

| Component | Role | License |
|---|---|---|
| MuseTalk (TMElyralab) | lip-sync UNet + VAE | MIT |
| sd-vae-ft-mse (Stability AI) | VAE | MIT |
| Whisper (OpenAI) | audio feature extraction | MIT |
| face-alignment / 2D-FAN-4 (1adrianb) | facial landmarks for the blend mask | BSD-3-Clause |
| Stable Diffusion v1.4 UNet (CompVis / Stability AI) | MuseTalk fine-tune base | CreativeML Open RAIL-M |

The face blend mask is derived from FAN facial landmarks.

Runtime GPU libraries — cuBLAS / cuDNN (CUDA), MPSGraph (Metal), RADV / Mesa (Vulkan) —
are linked at runtime (not redistributed). `ffmpeg` (binary) is used for muxing only.
