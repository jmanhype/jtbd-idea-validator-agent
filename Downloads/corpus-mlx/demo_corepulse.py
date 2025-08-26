#!/usr/bin/env python3
"""
Demo script showing all CorePulse features working with MLX Stable Diffusion.
Uses our fixed wrapper and shows visual results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper_fixed import CorePulseStableDiffusion

def decode_and_save(sd, latents, filename):
    """Decode latents and save as image"""
    # Decode latents to images
    images = sd.autoencoder.decode(latents)
    images = mx.clip(images / 2 + 0.5, 0, 1)
    images = (images * 255).astype(mx.uint8)
    
    # Convert to numpy and PIL
    images_np = np.array(images)
    if images_np.ndim == 4:
        images_np = images_np[0]  # First image
    # Ensure HWC format
    if images_np.shape[0] in [3, 4]:
        images_np = np.transpose(images_np, (1, 2, 0))
    
    img = Image.fromarray(images_np)
    img.save(filename)
    print(f"✓ Saved: {filename}")
    return img

def demo_prompt_injection():
    """Demo 1: Time-windowed prompt injection"""
    print("\n=== DEMO 1: PROMPT INJECTION ===")
    print("Injecting different prompts at different time windows...")
    
    # Load model
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    wrapper = CorePulseStableDiffusion(sd)
    
    # Add time-windowed injections
    wrapper.add_injection(
        prompt="abstract geometric patterns",
        start_frac=0.0,
        end_frac=0.3,
        weight=0.8
    )
    
    wrapper.add_injection(
        prompt="vibrant neon colors",
        start_frac=0.3,
        end_frac=0.7,
        weight=0.6
    )
    
    wrapper.add_injection(
        prompt="intricate fine details",
        start_frac=0.7,
        end_frac=1.0,
        weight=0.4
    )
    
    # Generate
    print("Generating with time-windowed injections...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "digital art masterpiece",
        num_steps=25,
        cfg_weight=7.5,
        seed=42
    ):
        latents = step_latents
    
    decode_and_save(sd, latents, "demo1_prompt_injection.png")
    print("✓ Time-windowed injection complete!")

def demo_token_masking():
    """Demo 2: Token-level attention masking"""
    print("\n=== DEMO 2: TOKEN MASKING ===")
    print("Focusing on specific tokens in the prompt...")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    wrapper = CorePulseStableDiffusion(sd)
    
    # Add injection with token masking
    wrapper.add_injection(
        prompt="a majestic golden eagle soaring through clouds",
        token_mask="golden eagle",  # Focus only on these tokens
        weight=0.9
    )
    
    # Generate
    print("Generating with token masking...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "a bird in the sky",
        num_steps=25,
        cfg_weight=7.5,
        seed=42
    ):
        latents = step_latents
    
    decode_and_save(sd, latents, "demo2_token_masking.png")
    print("✓ Token masking complete!")

def demo_regional_control():
    """Demo 3: Regional/spatial prompt control"""
    print("\n=== DEMO 3: REGIONAL CONTROL ===")
    print("Different prompts for different image regions...")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    wrapper = CorePulseStableDiffusion(sd)
    
    # Left half: ocean
    wrapper.add_injection(
        prompt="deep blue ocean waves",
        region=("rect_pix", 0, 0, 256, 512, 10),  # Left half with 10px feather
        weight=0.8
    )
    
    # Right half: desert
    wrapper.add_injection(
        prompt="golden sand dunes",
        region=("rect_pix", 256, 0, 256, 512, 10),  # Right half with 10px feather
        weight=0.8
    )
    
    # Generate
    print("Generating with regional control...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "landscape photography",
        num_steps=25,
        cfg_weight=7.5,
        seed=42
    ):
        latents = step_latents
    
    decode_and_save(sd, latents, "demo3_regional_control.png")
    print("✓ Regional control complete!")

def demo_combined():
    """Demo 4: All features combined"""
    print("\n=== DEMO 4: COMBINED FEATURES ===")
    print("Using all CorePulse features together...")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    wrapper = CorePulseStableDiffusion(sd)
    
    # Time-windowed injection
    wrapper.add_injection(
        prompt="cyberpunk aesthetic",
        start_frac=0.0,
        end_frac=0.4,
        weight=0.6
    )
    
    # Token-masked injection
    wrapper.add_injection(
        prompt="neon lights glowing cityscape",
        token_mask="neon lights",
        weight=0.7
    )
    
    # Regional injection
    wrapper.add_injection(
        prompt="futuristic skyscrapers",
        region=("rect_pix", 100, 100, 312, 312, 15),  # Center region with feather
        weight=0.8
    )
    
    # Generate
    print("Generating with all features...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "night city scene",
        negative_text="blurry, low quality",
        num_steps=30,
        cfg_weight=8.0,
        seed=42
    ):
        latents = step_latents
    
    decode_and_save(sd, latents, "demo4_combined.png")
    print("✓ Combined features complete!")

def main():
    """Run all CorePulse demos"""
    print("=" * 60)
    print("COREPULSE MLX DEMO")
    print("Demonstrating advanced prompt injection features")
    print("=" * 60)
    
    try:
        # Run demos
        print("\n[1/2] Running prompt injection demo...")
        demo_prompt_injection()
        print("\n[2/2] Running regional control demo...")
        demo_regional_control()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED!")
        print("Generated images:")
        print("  - demo1_prompt_injection.png")
        print("  - demo2_token_masking.png")
        print("  - demo3_regional_control.png")
        print("  - demo4_combined.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())