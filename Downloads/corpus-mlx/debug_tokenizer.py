#!/usr/bin/env python3
import mlx.core as mx
from stable_diffusion import StableDiffusion

def main():
    print("=== Testing Tokenization Differences ===\n")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    test_prompt = "a beautiful mountain landscape"
    
    # Test original tokenization method
    print("1. Original SD tokenization (_tokenize method):")
    tokens_orig = sd._tokenize(sd.tokenizer, test_prompt)
    print(f"   Shape: {tokens_orig.shape}")
    print(f"   Tokens: {tokens_orig}")
    print(f"   First few values: {tokens_orig[0][:10]}")
    
    # Test direct tokenizer.tokenize
    print("\n2. Direct tokenizer.tokenize:")
    tokens_direct = sd.tokenizer.tokenize(test_prompt)
    print(f"   Type: {type(tokens_direct)}")
    print(f"   Length: {len(tokens_direct)}")
    print(f"   Tokens: {tokens_direct}")
    
    # Test our encode_tokens method
    from corpus_mlx.injection import encode_tokens
    print("\n3. Our encode_tokens method:")
    tokens_ours = encode_tokens(sd, test_prompt)
    print(f"   Shape: {tokens_ours.shape}")
    print(f"   Tokens: {tokens_ours}")
    
    # Test text encoding
    print("\n4. Text encoding comparison:")
    
    # Original method
    conditioning_orig = sd._get_text_conditioning(test_prompt, n_images=1, cfg_weight=7.5)
    print(f"   Original conditioning shape: {conditioning_orig.shape}")
    print(f"   Original conditioning dtype: {conditioning_orig.dtype}")
    
    # Our method
    emb_ours = sd.text_encoder(tokens_ours).last_hidden_state
    print(f"   Our embedding shape: {emb_ours.shape}")
    print(f"   Our embedding dtype: {emb_ours.dtype}")
    
    # Check if they're similar
    if conditioning_orig.shape == emb_ours.shape:
        diff = mx.abs(conditioning_orig - emb_ours).max()
        print(f"   Max difference: {diff}")
    else:
        print("   ERROR: Shape mismatch!")

if __name__ == "__main__":
    main()