#!/usr/bin/env python3
import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper_fixed import CorePulseStableDiffusion

def generate_image(cpsd, sd, prompt, filename, seed=42):
    print(f"\nGenerating: '{prompt}'")
    latents = cpsd.generate_latents(
        base_prompt=prompt,
        negative_text="low quality, blurry, distorted",
        num_steps=15, cfg_weight=7.5,
        n_images=1, height=512, width=512, seed=seed
    )
    
    for x_t in latents: 
        mx.eval(x_t)
    
    img = sd.decode(x_t)
    mx.eval(img)
    
    # Save image
    img_np = np.array(img)
    img_np = (img_np * 255).astype(np.uint8)
    im = Image.fromarray(img_np[0])
    im.save(filename)
    print(f"✓ Saved to {filename}")

def main():
    print("Loading Stable Diffusion model...")
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    cpsd = CorePulseStableDiffusion(sd)

    # Test different prompts
    test_cases = [
        ("a serene lake with mountains", "fixed_lake.png", 100),
        ("a futuristic city skyline", "fixed_city.png", 200),
        ("a cute cat wearing sunglasses", "fixed_cat.png", 300),
        ("a tropical beach at sunset", "fixed_beach.png", 400),
    ]
    
    for prompt, filename, seed in test_cases:
        generate_image(cpsd, sd, prompt, filename, seed)
    
    print("\n✅ All test images generated successfully!")

if __name__ == "__main__":
    main()