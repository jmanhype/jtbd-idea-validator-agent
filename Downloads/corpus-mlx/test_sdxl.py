#!/usr/bin/env python3
"""
Test script for Stable Diffusion XL with MLX
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from stable_diffusion import StableDiffusionXL


def save_image(image_array, filename):
    """Save MLX array as image."""
    # Convert to numpy and proper format
    image_np = np.array(image_array)
    
    # Handle different array shapes
    if image_np.ndim == 4:
        image_np = image_np[0]  # Take first image if batch
    
    # Convert from CHW to HWC if needed
    if image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))
    
    # Ensure values are in [0, 255]
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    
    # Save image
    image = Image.fromarray(image_np)
    image.save(filename)
    print(f"Image saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate images using SDXL with MLX")
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/sdxl-turbo",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A majestic lion in a sunset landscape, highly detailed, 8k resolution",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative text prompt",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=4,  # SDXL Turbo works well with fewer steps
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=0.0,  # SDXL Turbo is trained without CFG
        help="CFG weight for guidance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sdxl_output.png",
        help="Output image filename",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (SDXL default is 1024)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (SDXL default is 1024)",
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Use float16 precision",
    )
    
    args = parser.parse_args()
    
    # Calculate latent dimensions (divide by 8 for VAE)
    latent_height = args.height // 8
    latent_width = args.width // 8
    
    print(f"Loading SDXL model: {args.model}")
    print(f"Using {'float16' if args.float16 else 'float32'} precision")
    
    # Initialize SDXL model
    sdxl = StableDiffusionXL(
        model=args.model,
        float16=args.float16
    )
    
    print("Ensuring models are loaded...")
    sdxl.ensure_models_are_loaded()
    
    print(f"\nGenerating image with prompt: '{args.prompt}'")
    print(f"Image size: {args.width}x{args.height}")
    print(f"Steps: {args.num_steps}, CFG: {args.cfg}")
    
    # Set seed if provided
    if args.seed is not None:
        mx.random.seed(args.seed)
        print(f"Using seed: {args.seed}")
    
    # Generate latents
    start_time = time.time()
    
    latents = None
    for step, x_t in enumerate(sdxl.generate_latents(
        text=args.prompt,
        n_images=1,
        num_steps=args.num_steps,
        cfg_weight=args.cfg,
        negative_text=args.negative_prompt,
        latent_size=(latent_height, latent_width),
        seed=args.seed,
    )):
        latents = x_t
        mx.eval(latents)
        print(f"Step {step + 1}/{args.num_steps} completed")
    
    print("\nDecoding latents to image...")
    # Decode the latents to an image
    images = sdxl.decode(latents)
    mx.eval(images)
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Save the image
    save_image(images, args.output)
    
    print(f"\nSDXL generation successful!")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()