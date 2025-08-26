#!/usr/bin/env python3
import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper_fixed import CorePulseStableDiffusion

def main():
    print("Loading Stable Diffusion model...")
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    cpsd = CorePulseStableDiffusion(sd)

    print("Generating: 'a beautiful mountain landscape'")
    latents = cpsd.generate_latents(
        base_prompt="a beautiful mountain landscape",
        negative_text="low quality, blurry",
        num_steps=10, cfg_weight=7.5,
        n_images=1, height=512, width=512, seed=42
    )
    
    for x_t in latents: 
        mx.eval(x_t)
    
    print("Decoding image...")
    img = sd.decode(x_t)
    mx.eval(img)
    
    # Save image using PIL
    img_np = np.array(img)
    img_np = (img_np * 255).astype(np.uint8)
    im = Image.fromarray(img_np[0])  # Take first image
    im.save("test_fixed.png")
    print("Image saved to test_fixed.png")

if __name__ == "__main__":
    main()