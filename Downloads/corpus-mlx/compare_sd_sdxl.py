#!/usr/bin/env python3
"""
Compare Stable Diffusion 2.1 vs SDXL outputs
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from stable_diffusion import StableDiffusion, StableDiffusionXL


def save_image(image_array, filename):
    """Save MLX array as image."""
    image_np = np.array(image_array)
    
    if image_np.ndim == 4:
        image_np = image_np[0]
    
    if image_np.shape[0] in [3, 4]:
        image_np = np.transpose(image_np, (1, 2, 0))
    
    image_np = (image_np * 255).astype(np.uint8)
    image = Image.fromarray(image_np)
    image.save(filename)
    return image


def generate_with_sd(prompt, negative_prompt="", seed=None, num_steps=50, cfg=7.5):
    """Generate image with Stable Diffusion 2.1"""
    print("\n" + "="*50)
    print("Generating with Stable Diffusion 2.1")
    print("="*50)
    
    sd = StableDiffusion(
        model="stabilityai/stable-diffusion-2-1-base",
        float16=True
    )
    
    sd.ensure_models_are_loaded()
    
    if seed is not None:
        mx.random.seed(seed)
    
    start_time = time.time()
    
    latents = None
    for step, x_t in enumerate(sd.generate_latents(
        text=prompt,
        n_images=1,
        num_steps=num_steps,
        cfg_weight=cfg,
        negative_text=negative_prompt,
        latent_size=(64, 64),  # SD 2.1 uses 512x512 by default
        seed=seed,
    )):
        latents = x_t
        mx.eval(latents)
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{num_steps}")
    
    images = sd.decode(latents)
    mx.eval(images)
    
    generation_time = time.time() - start_time
    print(f"SD 2.1 generation time: {generation_time:.2f} seconds")
    
    return images, generation_time


def generate_with_sdxl(prompt, negative_prompt="", seed=None, num_steps=4, cfg=0.0):
    """Generate image with SDXL Turbo"""
    print("\n" + "="*50)
    print("Generating with SDXL Turbo")
    print("="*50)
    
    sdxl = StableDiffusionXL(
        model="stabilityai/sdxl-turbo",
        float16=True
    )
    
    sdxl.ensure_models_are_loaded()
    
    if seed is not None:
        mx.random.seed(seed)
    
    start_time = time.time()
    
    latents = None
    for step, x_t in enumerate(sdxl.generate_latents(
        text=prompt,
        n_images=1,
        num_steps=num_steps,
        cfg_weight=cfg,
        negative_text=negative_prompt,
        latent_size=(128, 128),  # SDXL uses 1024x1024 by default
        seed=seed,
    )):
        latents = x_t
        mx.eval(latents)
        print(f"Step {step + 1}/{num_steps}")
    
    images = sdxl.decode(latents)
    mx.eval(images)
    
    generation_time = time.time() - start_time
    print(f"SDXL Turbo generation time: {generation_time:.2f} seconds")
    
    return images, generation_time


def create_comparison_image(sd_image, sdxl_image, prompt, sd_time, sdxl_time):
    """Create a side-by-side comparison image"""
    # Convert MLX arrays to PIL images
    sd_pil = Image.fromarray((np.array(sd_image[0]).transpose(1, 2, 0) * 255).astype(np.uint8))
    sdxl_pil = Image.fromarray((np.array(sdxl_image[0]).transpose(1, 2, 0) * 255).astype(np.uint8))
    
    # Resize SDXL to match SD height for comparison
    sdxl_pil_resized = sdxl_pil.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Create comparison image
    comparison = Image.new('RGB', (1024 + 20, 512 + 100))
    comparison.paste(sd_pil, (0, 50))
    comparison.paste(sdxl_pil_resized, (512 + 20, 50))
    
    # Add labels using PIL's built-in font (basic)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(comparison)
    
    # Add title
    draw.text((10, 10), f"Prompt: {prompt[:80]}...", fill="white")
    
    # Add model labels
    draw.text((10, 30), f"SD 2.1 ({sd_time:.1f}s)", fill="white")
    draw.text((532, 30), f"SDXL Turbo ({sdxl_time:.1f}s)", fill="white")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Compare SD 2.1 vs SDXL")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A futuristic city with flying cars, neon lights, cyberpunk style, highly detailed",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for consistent comparison",
    )
    parser.add_argument(
        "--sd-steps",
        type=int,
        default=50,
        help="Number of steps for SD 2.1",
    )
    parser.add_argument(
        "--sdxl-steps",
        type=int,
        default=4,
        help="Number of steps for SDXL Turbo",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./comparison_output",
        help="Output directory for images",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Comparing SD 2.1 vs SDXL Turbo")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")
    
    # Generate with SD 2.1
    sd_images, sd_time = generate_with_sd(
        args.prompt,
        seed=args.seed,
        num_steps=args.sd_steps,
        cfg=7.5
    )
    sd_image_pil = save_image(sd_images, output_dir / "sd21_output.png")
    
    # Generate with SDXL Turbo
    sdxl_images, sdxl_time = generate_with_sdxl(
        args.prompt,
        seed=args.seed,
        num_steps=args.sdxl_steps,
        cfg=0.0  # SDXL Turbo doesn't use CFG
    )
    sdxl_image_pil = save_image(sdxl_images, output_dir / "sdxl_output.png")
    
    # Create comparison
    comparison = create_comparison_image(
        sd_images, sdxl_images, args.prompt, sd_time, sdxl_time
    )
    comparison.save(output_dir / "comparison.png")
    
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"SD 2.1: {sd_time:.2f} seconds ({args.sd_steps} steps)")
    print(f"SDXL Turbo: {sdxl_time:.2f} seconds ({args.sdxl_steps} steps)")
    print(f"Speedup: {sd_time/sdxl_time:.2f}x")
    print(f"\nImages saved to {output_dir}/")
    print("  - sd21_output.png: Stable Diffusion 2.1 output")
    print("  - sdxl_output.png: SDXL Turbo output")  
    print("  - comparison.png: Side-by-side comparison")


if __name__ == "__main__":
    main()