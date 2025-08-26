#!/usr/bin/env python3
import mlx.core as mx
from stable_diffusion import StableDiffusion
import time

def main():
    print("=== Testing UNet Call ===\n")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    test_prompt = "a beautiful mountain landscape"
    negative_prompt = ""
    
    # Get proper conditioning
    conditioning = sd._get_text_conditioning(
        test_prompt, 
        n_images=1, 
        cfg_weight=7.5,
        negative_text=negative_prompt
    )
    print(f"Conditioning shape: {conditioning.shape}")
    
    # Create test latent
    latent_size = (64, 64)
    mx.random.seed(42)
    x_t = sd.sampler.sample_prior(
        (1, *latent_size, sd.autoencoder.latent_channels), 
        dtype=mx.float16
    )
    print(f"Latent shape: {x_t.shape}")
    
    # Get a timestep
    t = mx.array([1000.0])
    
    # Test how the original calls UNet in _denoising_step
    print("\n1. Original _denoising_step pattern:")
    
    # When cfg_weight > 1, it concatenates latents too!
    x_t_unet = mx.concatenate([x_t] * 2, axis=0)  # Duplicate for CFG
    t_unet = mx.broadcast_to(t, [len(x_t_unet)])
    
    print(f"   x_t_unet shape: {x_t_unet.shape}")
    print(f"   t_unet shape: {t_unet.shape}")
    print(f"   conditioning shape: {conditioning.shape}")
    
    # Call UNet
    print("\n2. Calling UNet:")
    eps_pred = sd.unet(x_t_unet, t_unet, encoder_x=conditioning)
    print(f"   eps_pred shape: {eps_pred.shape}")
    
    # Split the predictions
    eps_text, eps_neg = eps_pred.split(2)
    print(f"   eps_text shape: {eps_text.shape}")
    print(f"   eps_neg shape: {eps_neg.shape}")
    
    # Apply CFG
    cfg_weight = 7.5
    eps_final = eps_neg + cfg_weight * (eps_text - eps_neg)
    print(f"   eps_final shape: {eps_final.shape}")

if __name__ == "__main__":
    main()