#!/usr/bin/env python3
"""Quick test of regional control feature"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper_fixed import CorePulseStableDiffusion

print("Testing Regional Control...")

# Load model
sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
wrapper = CorePulseStableDiffusion(sd)

# Left half: fire
wrapper.add_injection(
    prompt="blazing fire flames orange red",
    region=("rect_pix", 0, 0, 256, 512, 20),  # Left half
    weight=0.9
)

# Right half: ice
wrapper.add_injection(
    prompt="frozen ice crystals blue white",
    region=("rect_pix", 256, 0, 256, 512, 20),  # Right half
    weight=0.9
)

# Generate with fewer steps for speed
print("Generating fire & ice split image...")
latents = None
for step_latents in wrapper.generate_latents(
    "elemental contrast",
    num_steps=15,  # Fewer steps for quick test
    cfg_weight=7.5,
    seed=42
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
img.save("test_regional_fire_ice.png")
print("✓ Saved: test_regional_fire_ice.png")
print("✓ Regional control test complete!")