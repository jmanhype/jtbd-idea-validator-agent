#!/usr/bin/env python3
import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper_simple import CorePulseStableDiffusion

def main():
    print("Loading Stable Diffusion model...")
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    cpsd = CorePulseStableDiffusion(sd)

    # Add some test injections (these won't work yet since injection logic isn't implemented)
    print("Adding injection configurations...")
    cpsd.add_injection(
        prompt="a bright sun",
        weight=0.85,
        start_frac=0.6, end_frac=1.0,
        token_mask="sun",
        region=("rect_frac", 0.65, 0.05, 0.32, 0.32, 0.10),
    )

    print("Generating image with injections (fallback to base generation)...")
    latents = cpsd.generate_latents(
        base_prompt="a mountain landscape at sunrise",
        negative_text="low quality, blurry",
        num_steps=20, cfg_weight=7.0,
        n_images=1, height=512, width=512, seed=123
    )
    
    for x_t in latents: 
        mx.eval(x_t)
    
    print("Decoding image...")
    img = sd.decode(x_t); mx.eval(img)
    
    # Save image using PIL
    img_np = np.array(img)
    img_np = (img_np * 255).astype(np.uint8)
    im = Image.fromarray(img_np[0])  # Take first image
    im.save("demo_injection.png")
    print("Image saved to demo_injection.png")
    print("Note: Injection logic is not yet implemented, this shows base functionality.")

if __name__ == "__main__":
    main()