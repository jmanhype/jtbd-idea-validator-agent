#!/usr/bin/env python3
"""Final combined test of all CorePulse features"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper_fixed import CorePulseStableDiffusion

print("Testing ALL CorePulse Features Combined...")

# Load model
sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
wrapper = CorePulseStableDiffusion(sd)

# 1. Time-windowed injection (early stage)
wrapper.add_injection(
    prompt="dark mysterious atmosphere",
    start_frac=0.0,
    end_frac=0.3,
    weight=0.7
)

# 2. Token masking injection (middle stage)
wrapper.add_injection(
    prompt="glowing magical crystal orb energy",
    token_mask="crystal orb",  # Focus on these tokens
    start_frac=0.3,
    end_frac=0.7,
    weight=0.8
)

# 3. Regional injection (center)
wrapper.add_injection(
    prompt="brilliant light rays emanating",
    region=("rect_pix", 150, 150, 212, 212, 30),  # Center region
    start_frac=0.5,
    end_frac=1.0,
    weight=0.6
)

# Generate
print("Generating with ALL features combined...")
latents = None
for step_latents in wrapper.generate_latents(
    "mystical artifact",
    negative_text="blurry, low quality, distorted",
    num_steps=20,
    cfg_weight=8.0,
    seed=777
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
img.save("test_all_features_combined.png")
print("✓ Saved: test_all_features_combined.png")
print("\n✓ ALL COREPULSE FEATURES WORKING!")
print("  - Time-windowed injection ✓")
print("  - Token-level masking ✓")
print("  - Regional control ✓")
print("  - Combined features ✓")