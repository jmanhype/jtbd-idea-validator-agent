#!/usr/bin/env python3
"""Validation test for advanced features with clear visual feedback."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper_fixed import CorePulseStableDiffusion

def save_image(latents, sd, filename):
    """Helper to decode and save image."""
    images = sd.autoencoder.decode(latents)
    images = mx.clip(images / 2 + 0.5, 0, 1)
    images = (images * 255).astype(mx.uint8)
    
    images_np = np.array(images)
    if images_np.ndim == 4:
        images_np = images_np[0]
    if images_np.shape[0] in [3, 4]:
        images_np = np.transpose(images_np, (1, 2, 0))
    
    img = Image.fromarray(images_np)
    img.save(filename)
    print(f"✓ Saved: {filename}")
    return img

print("VALIDATION TEST - Advanced Features")
print("="*50)

# Load model
sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
wrapper = CorePulseStableDiffusion(sd)

# Test 1: Clear attention control - early vs late
print("\n1. Testing attention control (time-based)...")
wrapper.clear_injections()

# Early: Red theme
wrapper.add_injection(
    prompt="bright red crimson scarlet ruby",
    start_frac=0.0,
    end_frac=0.5,
    weight=0.9
)

# Late: Blue theme  
wrapper.add_injection(
    prompt="deep blue azure navy sapphire",
    start_frac=0.5,
    end_frac=1.0,
    weight=0.9
)

latents = None
for step_latents in wrapper.generate_latents(
    "geometric shapes",
    negative_text="green, yellow, purple",
    num_steps=15,
    cfg_weight=9.0,
    seed=1111
):
    latents = step_latents

save_image(latents, sd, "validate_1_attention.png")
print("Expected: Red-to-blue color transition in shapes")

# Test 2: Token focus test
print("\n2. Testing token-level focus...")
wrapper.clear_injections()

wrapper.add_injection(
    prompt="golden crown jeweled throne royal scepter",
    token_mask="crown throne",  # Focus only on crown and throne
    weight=0.95
)

latents = None
for step_latents in wrapper.generate_latents(
    "royal chamber",
    num_steps=15,
    cfg_weight=8.5,
    seed=2222
):
    latents = step_latents

save_image(latents, sd, "validate_2_tokens.png")
print("Expected: Strong emphasis on crown and throne")

# Test 3: Clear regional test - split screen
print("\n3. Testing regional control (split)...")
wrapper.clear_injections()

# Left half: Fire
wrapper.add_injection(
    prompt="blazing fire flames orange red hot lava",
    region=("rect_pix", 0, 0, 256, 512, 5),
    weight=0.9
)

# Right half: Ice
wrapper.add_injection(
    prompt="frozen ice crystal blue cold snow frost",
    region=("rect_pix", 256, 0, 256, 512, 5),
    weight=0.9
)

latents = None
for step_latents in wrapper.generate_latents(
    "elemental forces",
    negative_text="green, purple",
    num_steps=15,
    cfg_weight=8.0,
    seed=3333
):
    latents = step_latents

save_image(latents, sd, "validate_3_regions.png")
print("Expected: Fire on left, ice on right")

# Test 4: Multi-scale progression
print("\n4. Testing multi-scale (coarse to fine)...")
wrapper.clear_injections()

# Stage 1: Big circles (early)
wrapper.add_injection(
    prompt="large circles spheres bubbles",
    start_frac=0.0,
    end_frac=0.4,
    weight=0.9
)

# Stage 2: Add squares (middle)
wrapper.add_injection(
    prompt="squares rectangles boxes geometric",
    start_frac=0.3,
    end_frac=0.7,
    weight=0.8
)

# Stage 3: Add fine details (late)
wrapper.add_injection(
    prompt="tiny dots stippling texture grain detail",
    start_frac=0.6,
    end_frac=1.0,
    weight=0.7
)

latents = None
for step_latents in wrapper.generate_latents(
    "abstract composition",
    num_steps=20,
    cfg_weight=8.0,
    seed=4444
):
    latents = step_latents

save_image(latents, sd, "validate_4_multiscale.png")
print("Expected: Circles → squares → fine texture progression")

# Test 5: Combined test - all features
print("\n5. Testing ALL features combined...")
wrapper.clear_injections()

# Base atmosphere (early)
wrapper.add_injection(
    prompt="dark mysterious foggy",
    start_frac=0.0,
    end_frac=0.3,
    weight=0.7
)

# Center glow (regional + token)
wrapper.add_injection(
    prompt="bright glowing orb crystal sphere luminous",
    token_mask="orb crystal",
    region=("circle_pix", 256, 256, 100, 30),
    start_frac=0.2,
    end_frac=0.9,
    weight=0.85
)

# Corner details (late + regional)
wrapper.add_injection(
    prompt="electric sparks lightning",
    region=("rect_pix", 0, 0, 150, 150, 20),
    start_frac=0.6,
    end_frac=1.0,
    weight=0.6
)

latents = None
for step_latents in wrapper.generate_latents(
    "mystical portal",
    negative_text="daylight, bright, cartoon",
    num_steps=20,
    cfg_weight=8.5,
    seed=5555
):
    latents = step_latents

save_image(latents, sd, "validate_5_combined.png")
print("Expected: Dark scene with glowing center orb and corner sparks")

print("\n" + "="*50)
print("VALIDATION COMPLETE!")
print("\nCheck images for:")
print("1. validate_1_attention.png - Red→Blue transition")
print("2. validate_2_tokens.png - Crown/throne focus")
print("3. validate_3_regions.png - Fire|Ice split")
print("4. validate_4_multiscale.png - Progressive detail")
print("5. validate_5_combined.png - All features")