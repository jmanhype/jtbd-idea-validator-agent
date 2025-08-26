#!/usr/bin/env python3
"""
Quick test to verify SDXL is working
"""

import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusionXL

def test_sdxl():
    print("Initializing SDXL Turbo...")
    
    # Initialize with SDXL Turbo (fast model)
    sdxl = StableDiffusionXL(
        model="stabilityai/sdxl-turbo",
        float16=True  # Use float16 for speed
    )
    
    print("Loading models...")
    sdxl.ensure_models_are_loaded()
    
    # Simple prompt
    prompt = "A red apple on a white table"
    
    print(f"\nGenerating image: '{prompt}'")
    print("Using SDXL Turbo with 1 step (ultra-fast mode)")
    
    # Generate with minimal steps for quick test
    latents = None
    for x_t in sdxl.generate_latents(
        text=prompt,
        n_images=1,
        num_steps=1,  # Just 1 step for ultra-fast generation
        cfg_weight=0.0,  # SDXL Turbo doesn't need CFG
        latent_size=(128, 128),  # 1024x1024 output
    ):
        latents = x_t
        mx.eval(latents)
    
    print("Decoding image...")
    images = sdxl.decode(latents)
    mx.eval(images)
    
    # Convert and save
    image_np = np.array(images)
    
    # Handle batch dimension
    if image_np.ndim == 4:
        image_np = image_np[0]
    
    # Ensure CHW to HWC conversion
    if image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))
    
    # Clip values and convert to uint8
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    
    image = Image.fromarray(image_np)
    image.save("quick_sdxl_test.png")
    
    print("\n✅ SDXL test successful!")
    print("Image saved to: quick_sdxl_test.png")
    print(f"Image size: {image.size}")
    
    return True

if __name__ == "__main__":
    try:
        test_sdxl()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis might be because the model needs to be downloaded.")
        print("The first run will download the model from Hugging Face.")
        raise