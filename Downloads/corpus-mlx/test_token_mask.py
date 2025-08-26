#!/usr/bin/env python3
"""Test token masking - emphasize specific words"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper_fixed import CorePulseStableDiffusion

print("Testing Token Masking...")

# Load model
sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
wrapper = CorePulseStableDiffusion(sd)

# Add injection focusing on "dragon" token only
wrapper.add_injection(
    prompt="majestic golden dragon breathing fire in clouds",
    token_mask="dragon",  # Focus only on dragon
    weight=0.95
)

# Generate
print("Generating with token masking on 'dragon'...")
latents = None
for step_latents in wrapper.generate_latents(
    "fantasy creature in sky",
    num_steps=15,
    cfg_weight=7.5,
    seed=123
):
    latents = step_latents

# Decode and save
images = sd.autoencoder.decode(latents)
images = mx.clip(images / 2 + 0.5, 0, 1)
images = (images * 255).astype(mx.uint8)

images_np = np.array(images)
if images_np.ndim == 4:
    images_np = images_np[0]
if images_np.shape[0] in [3, 4]:
    images_np = np.transpose(images_np, (1, 2, 0))

img = Image.fromarray(images_np)
img.save("test_token_mask_dragon.png")
print("✓ Saved: test_token_mask_dragon.png")
print("✓ Token masking test complete!")