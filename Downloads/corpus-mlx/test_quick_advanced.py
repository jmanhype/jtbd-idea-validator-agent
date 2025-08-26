#!/usr/bin/env python3
"""Quick test of each advanced feature."""

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

print("QUICK ADVANCED FEATURES TEST")
print("="*40)

# Load model
sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
wrapper = CorePulseStableDiffusion(sd)

# Test 1: Attention manipulation simulation
print("\n1. Attention manipulation (via time windows)...")
wrapper.clear_injections()
wrapper.add_injection(prompt="bold geometric", start_frac=0.0, end_frac=0.4, weight=0.9)
wrapper.add_injection(prompt="soft organic", start_frac=0.6, end_frac=1.0, weight=0.7)

latents = None
for step_latents in wrapper.generate_latents("abstract", num_steps=10, seed=111):
    latents = step_latents
save_image(latents, sd, "quick_attention.png")

# Test 2: Cross-attention redistribution (via token masking)
print("\n2. Cross-attention redistribution...")
wrapper.clear_injections()
wrapper.add_injection(
    prompt="majestic eagle soaring wings spread",
    token_mask="eagle wings",  # Focus attention on these tokens
    weight=0.9
)

latents = None
for step_latents in wrapper.generate_latents("bird", num_steps=10, seed=222):
    latents = step_latents
save_image(latents, sd, "quick_redistribution.png")

# Test 3: Per-block control (via regions simulating blocks)
print("\n3. Per-block control simulation...")
wrapper.clear_injections()
# Simulate different blocks with regions
wrapper.add_injection(prompt="sky", region=("rect_pix", 0, 0, 512, 200, 20), weight=0.8)
wrapper.add_injection(prompt="mountains", region=("rect_pix", 0, 200, 512, 150, 20), weight=0.8)
wrapper.add_injection(prompt="lake", region=("rect_pix", 0, 350, 512, 162, 20), weight=0.8)

latents = None
for step_latents in wrapper.generate_latents("landscape", num_steps=10, seed=333):
    latents = step_latents
save_image(latents, sd, "quick_perblock.png")

# Test 4: Multi-scale (coarse to fine)
print("\n4. Multi-scale control...")
wrapper.clear_injections()
wrapper.add_injection(prompt="large shapes", start_frac=0.0, end_frac=0.4, weight=0.8)
wrapper.add_injection(prompt="medium details", start_frac=0.3, end_frac=0.7, weight=0.7)
wrapper.add_injection(prompt="tiny intricate", start_frac=0.6, end_frac=1.0, weight=0.6)

latents = None
for step_latents in wrapper.generate_latents("fractal", num_steps=10, seed=444):
    latents = step_latents
save_image(latents, sd, "quick_multiscale.png")

print("\n" + "="*40)
print("✅ ADVANCED FEATURES DOGFOODING COMPLETE!")
print("\nGenerated test images:")
print("- quick_attention.png")
print("- quick_redistribution.png")  
print("- quick_perblock.png")
print("- quick_multiscale.png")