#!/usr/bin/env python3
"""
Test the fixed advanced features with proper time window balancing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper_enhanced import EnhancedCoreStableDiffusion

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

print("TESTING FIXED ADVANCED FEATURES")
print("="*50)

# Load model
sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
wrapper = EnhancedCoreStableDiffusion(sd)

# Test 1: Fixed time window balance - Red to Blue transition
print("\n1. Testing fixed time window balance...")
wrapper.clear_injections()

wrapper.add_injection(
    prompt="bright red crimson fire",
    start_frac=0.0,
    end_frac=0.5,
    weight=0.8
)

wrapper.add_injection(
    prompt="deep blue ocean water",
    start_frac=0.5,
    end_frac=1.0,
    weight=0.8
)

latents = None
for step_latents in wrapper.generate_latents(
    "abstract shapes",
    negative_text="green, yellow",
    num_steps=20,
    cfg_weight=8.0,
    seed=7777
):
    latents = step_latents

save_image(latents, sd, "fixed_1_timebalance.png")
print("Expected: Clear red-to-blue transition")

# Test 2: Per-block control test
print("\n2. Testing per-block injection control...")
wrapper.clear_injections()

# Early blocks: Structure
wrapper.add_injection(
    prompt="massive castle fortress",
    blocks=["down_0", "down_1"],
    weight=0.8
)

# Mid blocks: Details
wrapper.add_injection(
    prompt="intricate gothic architecture",
    blocks=["mid"],
    weight=0.7
)

# Late blocks: Atmosphere
wrapper.add_injection(
    prompt="foggy mysterious atmosphere",
    blocks=["up_2", "up_3"],
    weight=0.6
)

latents = None
for step_latents in wrapper.generate_latents(
    "medieval scene",
    num_steps=20,
    cfg_weight=7.5,
    seed=8888
):
    latents = step_latents

save_image(latents, sd, "fixed_2_perblock.png")
print("Expected: Castle with gothic details and fog")

# Test 3: Attention manipulation test
print("\n3. Testing attention manipulation...")
wrapper.clear_injections()

wrapper.add_injection(
    prompt="majestic dragon breathing fire",
    attention_scale={
        "down_0": 1.5,  # Boost early structure
        "down_1": 1.3,
        "mid": 1.0,
        "up_2": 0.8,    # Reduce late details
        "up_3": 0.7
    },
    weight=0.9
)

latents = None
for step_latents in wrapper.generate_latents(
    "fantasy creature",
    num_steps=20,
    cfg_weight=8.0,
    seed=9999
):
    latents = step_latents

save_image(latents, sd, "fixed_3_attention.png")
print("Expected: Dragon with emphasized structure")

# Test 4: Combined features test
print("\n4. Testing ALL features combined...")
wrapper.clear_injections()

# Time-based atmosphere
wrapper.add_injection(
    prompt="dark stormy night",
    start_frac=0.0,
    end_frac=0.4,
    weight=0.7
)

# Block-specific structure
wrapper.add_injection(
    prompt="towering lighthouse",
    blocks=["down_0", "down_1", "mid"],
    weight=0.8
)

# Regional with attention
wrapper.add_injection(
    prompt="bright beacon light rays",
    region=("circle_pix", 256, 150, 80, 30),
    attention_scale={"mid": 1.5, "up_0": 1.3},
    start_frac=0.3,
    end_frac=0.9,
    weight=0.85
)

# Token-focused details
wrapper.add_injection(
    prompt="crashing waves ocean foam spray",
    token_mask="waves foam",
    start_frac=0.5,
    end_frac=1.0,
    weight=0.6
)

latents = None
for step_latents in wrapper.generate_latents(
    "coastal scene",
    negative_text="sunny, daylight",
    num_steps=25,
    cfg_weight=8.5,
    seed=10000
):
    latents = step_latents

save_image(latents, sd, "fixed_4_combined.png")
print("Expected: Lighthouse with beacon in stormy night")

print("\n" + "="*50)
print("✅ FIXED FEATURES TEST COMPLETE!")
print("\nGenerated images:")
print("1. fixed_1_timebalance.png - Balanced time windows")
print("2. fixed_2_perblock.png - Per-block injection") 
print("3. fixed_3_attention.png - Attention manipulation")
print("4. fixed_4_combined.png - All features working together")