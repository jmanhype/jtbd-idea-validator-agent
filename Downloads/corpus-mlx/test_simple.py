#!/usr/bin/env python3
import mlx.core as mx
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper_simple import CorePulseStableDiffusion

def main():
    print("Loading Stable Diffusion model...")
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    cpsd = CorePulseStableDiffusion(sd)

    print("Generating image...")
    latents = cpsd.generate_latents(
        base_prompt="a beautiful landscape",
        negative_text="low quality",
        num_steps=10, cfg_weight=7.0,
        n_images=1, height=512, width=512, seed=42
    )
    
    for x_t in latents: 
        mx.eval(x_t)
    
    print("Decoding image...")
    img = sd.decode(x_t); mx.eval(img)
    
    try:
        from PIL import Image
        import numpy as np
        # Convert from MLX array to numpy and save as image
        img_np = np.array(img)
        img_np = (img_np * 255).astype(np.uint8)
        im = Image.fromarray(img_np[0])  # Take first image
        im.save("test_output.png")
        print("Image saved to test_output.png")
    except Exception as e:
        print(f"Could not save image: {e}")

if __name__ == "__main__":
    main()