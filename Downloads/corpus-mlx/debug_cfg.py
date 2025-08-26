#!/usr/bin/env python3
import mlx.core as mx
from stable_diffusion import StableDiffusion

def main():
    print("=== Testing CFG Conditioning ===\n")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    test_prompt = "a beautiful mountain landscape"
    negative_prompt = "low quality, blurry"
    
    # Test original CFG conditioning
    print("1. Original _get_text_conditioning with CFG:")
    conditioning = sd._get_text_conditioning(
        test_prompt, 
        n_images=1, 
        cfg_weight=7.5,  # cfg > 1 means use negative prompt
        negative_text=negative_prompt
    )
    print(f"   Conditioning shape: {conditioning.shape}")
    print(f"   Note: Shape[0]=2 means [negative, positive] embeddings concatenated")
    
    # Let's manually replicate this
    print("\n2. Manual replication:")
    tokens_pos = sd._tokenize(sd.tokenizer, test_prompt)
    tokens_neg = sd._tokenize(sd.tokenizer, negative_prompt)
    print(f"   Positive tokens shape: {tokens_pos.shape}")
    print(f"   Negative tokens shape: {tokens_neg.shape}")
    
    # Concatenate for CFG
    tokens_combined = mx.concatenate([tokens_neg, tokens_pos], axis=0)
    print(f"   Combined tokens shape: {tokens_combined.shape}")
    
    # Encode combined
    conditioning_manual = sd.text_encoder(tokens_combined).last_hidden_state
    print(f"   Manual conditioning shape: {conditioning_manual.shape}")
    
    # Compare
    diff = mx.abs(conditioning - conditioning_manual).max()
    print(f"   Max difference from original: {diff}")

if __name__ == "__main__":
    main()