#!/usr/bin/env python3
"""
Simple test of advanced features one by one.
"""

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

print("Testing Advanced Features with sd_wrapper_fixed...")

# Load model
sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
wrapper = CorePulseStableDiffusion(sd)

# Test 1: Attention-like effect through time windows
print("\n1. Testing attention-like control via time windows...")
wrapper.clear_injections()

# Early: Structure
wrapper.add_injection(
    prompt="blocky geometric shapes",
    start_frac=0.0,
    end_frac=0.3,
    weight=0.8
)

# Mid: Transformation
wrapper.add_injection(
    prompt="flowing organic curves",
    start_frac=0.3,
    end_frac=0.6,
    weight=0.7
)

# Late: Details
wrapper.add_injection(
    prompt="intricate crystalline patterns",
    start_frac=0.6,
    end_frac=1.0,
    weight=0.6
)

latents = None
for step_latents in wrapper.generate_latents(
    "abstract art",
    num_steps=20,
    cfg_weight=8.0,
    seed=100
):
    latents = step_latents

save_image(latents, sd, "test_adv_1_timewindows.png")

# Test 2: Simulated per-block control with regions
print("\n2. Testing per-block simulation with regions...")
wrapper.clear_injections()

# Top region (like early blocks)
wrapper.add_injection(
    prompt="cloudy sky atmosphere",
    region=("rect_pix", 0, 0, 512, 170, 30),
    weight=0.7
)

# Middle region (like mid blocks)
wrapper.add_injection(
    prompt="mountain ridges and peaks",
    region=("rect_pix", 0, 170, 512, 170, 30),
    weight=0.7
)

# Bottom region (like late blocks)
wrapper.add_injection(
    prompt="forest trees and vegetation",
    region=("rect_pix", 0, 340, 512, 172, 30),
    weight=0.7
)

latents = None
for step_latents in wrapper.generate_latents(
    "landscape vista",
    num_steps=20,
    cfg_weight=7.5,
    seed=200
):
    latents = step_latents

save_image(latents, sd, "test_adv_2_regions.png")

# Test 3: Multi-scale simulation
print("\n3. Testing multi-scale via progressive injection...")
wrapper.clear_injections()

# Coarse (early)
wrapper.add_injection(
    prompt="large circular form",
    start_frac=0.0,
    end_frac=0.35,
    weight=0.8
)

# Medium (middle)
wrapper.add_injection(
    prompt="radial symmetry patterns",
    start_frac=0.25,
    end_frac=0.65,
    weight=0.7
)

# Fine (late)
wrapper.add_injection(
    prompt="tiny intricate details filigree",
    start_frac=0.55,
    end_frac=1.0,
    weight=0.6
)

latents = None
for step_latents in wrapper.generate_latents(
    "mandala design",
    num_steps=25,
    cfg_weight=8.0,
    seed=300
):
    latents = step_latents

save_image(latents, sd, "test_adv_3_multiscale.png")

# Test 4: Attention redistribution simulation
print("\n4. Testing attention redistribution via masking...")
wrapper.clear_injections()

# Focus on main subject (simulate redistribution)
wrapper.add_injection(
    prompt="glowing powerful dragon breathing fire",
    token_mask="dragon fire",  # Focus attention here
    start_frac=0.2,
    end_frac=0.9,
    weight=0.85
)

# De-emphasize background
wrapper.add_injection(
    prompt="simple minimal background",
    start_frac=0.0,
    end_frac=0.3,
    weight=0.3
)

latents = None
for step_latents in wrapper.generate_latents(
    "dragon in sky",
    num_steps=20,
    cfg_weight=8.5,
    seed=400
):
    latents = step_latents

save_image(latents, sd, "test_adv_4_redistribution.png")

# Test 5: All features combined
print("\n5. Testing ALL features combined...")
wrapper.clear_injections()

# Layer 1: Base atmosphere (early)
wrapper.add_injection(
    prompt="cyberpunk neon atmosphere",
    start_frac=0.0,
    end_frac=0.4,
    weight=0.6
)

# Layer 2: Regional control (center focus)
wrapper.add_injection(
    prompt="glowing holographic interface",
    region=("circle_pix", 256, 256, 120, 40),
    start_frac=0.2,
    end_frac=0.8,
    weight=0.7
)

# Layer 3: Token focus (late details)
wrapper.add_injection(
    prompt="chrome metal reflections circuitry",
    token_mask="chrome circuitry",
    start_frac=0.5,
    end_frac=1.0,
    weight=0.6
)

# Layer 4: Corner accents
wrapper.add_injection(
    prompt="data streams flowing",
    region=("rect_pix", 0, 0, 100, 100, 20),
    weight=0.5
)

latents = None
for step_latents in wrapper.generate_latents(
    "futuristic terminal",
    negative_text="blurry, low quality",
    num_steps=30,
    cfg_weight=8.0,
    seed=500
):
    latents = step_latents

img = save_image(latents, sd, "test_adv_5_combined.png")

print("\n" + "="*60)
print("ADVANCED FEATURES VISUAL VALIDATION COMPLETE!")
print("="*60)
print("\nGenerated images:")
print("1. test_adv_1_timewindows.png - Multi-stage time control")
print("2. test_adv_2_regions.png - Per-block simulation via regions")
print("3. test_adv_3_multiscale.png - Multi-scale progression")
print("4. test_adv_4_redistribution.png - Attention focus simulation")
print("5. test_adv_5_combined.png - All features combined")
print("\n✅ All advanced features dogfooded successfully!")