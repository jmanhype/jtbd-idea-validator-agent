#!/usr/bin/env python3
"""
Test all CorePulse features with visual validation.
Demonstrates:
1. Prompt Injection
2. Token-Level Attention Masking
3. Regional/Spatial Injection
4. Attention Manipulation
5. Multi-Scale Control
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_stable_diffusion import StableDiffusion
from corpus_mlx.corepulse_enhanced import CorePulseEnhanced

def save_image(latents, name, sd=None):
    """Decode latents and save as image"""
    from corpus_mlx.utils_extended import latents_to_pil
    img = latents_to_pil(latents, sd)
    img.save(name)
    print(f"Saved: {name}")
    return img

def test_prompt_injection():
    """Test 1: Prompt Injection with time windows"""
    print("\n=== TEST 1: PROMPT INJECTION ===")
    print("Testing time-windowed prompt injection...")
    
    # Load model
    sd = StableDiffusion.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    wrapper = CorePulseEnhanced(sd)
    
    # Base prompt
    base_prompt = "a serene mountain landscape"
    
    # Add injection for early steps (coarse structure)
    wrapper.add_prompt_injection(
        prompt="dramatic stormy clouds",
        start=0.0,
        end=0.3,  # First 30% of steps
        weight=0.7
    )
    
    # Add injection for middle steps (details)
    wrapper.add_prompt_injection(
        prompt="golden sunset lighting",
        start=0.3,
        end=0.7,  # Middle 40% of steps
        weight=0.5
    )
    
    # Add injection for late steps (fine details)
    wrapper.add_prompt_injection(
        prompt="sharp detailed textures",
        start=0.7,
        end=1.0,  # Last 30% of steps
        weight=0.3
    )
    
    # Generate
    print("Generating with time-windowed injections...")
    latents = None
    for step_latents in wrapper.generate_latents(
        base_prompt,
        num_steps=30,
        cfg_weight=7.5,
        seed=42
    ):
        latents = step_latents
    
    save_image(latents[0], "test1_prompt_injection.png", sd)
    print("✓ Prompt injection test complete")
    return latents, sd

def test_token_masking():
    """Test 2: Token-Level Attention Masking"""
    print("\n=== TEST 2: TOKEN-LEVEL ATTENTION MASKING ===")
    print("Testing selective token emphasis...")
    
    sd = StableDiffusion.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    wrapper = CorePulseEnhanced(sd)
    
    # Add masked injection focusing on specific tokens
    wrapper.add_token_masked_injection(
        prompt="a red sports car on a city street",
        focus_tokens="red sports car",  # Only these tokens will be emphasized
        weight=0.8
    )
    
    # Generate
    print("Generating with token masking...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "a vehicle on a road",  # Base prompt is more generic
        num_steps=30,
        cfg_weight=7.5,
        seed=42
    ):
        latents = step_latents
    
    save_image(latents[0], "test2_token_masking.png", sd)
    print("✓ Token masking test complete")
    return latents, sd

def test_regional_injection():
    """Test 3: Regional/Spatial Injection"""
    print("\n=== TEST 3: REGIONAL/SPATIAL INJECTION ===")
    print("Testing region-specific prompt control...")
    
    sd = StableDiffusion.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    wrapper = CorePulseEnhanced(sd)
    
    # Left region: forest
    wrapper.add_regional_injection(
        prompt="dense green forest with tall trees",
        region=(0, 0, 256, 512),  # Left half
        weight=0.8
    )
    
    # Right region: desert
    wrapper.add_regional_injection(
        prompt="sandy desert with cacti",
        region=(256, 0, 512, 512),  # Right half
        weight=0.8
    )
    
    # Top region: sky (overlapping)
    wrapper.add_injection(
        prompt="dramatic cloudy sky",
        region=(0, 0, 512, 200),  # Top portion
        region_type="box",
        weight=0.5
    )
    
    # Generate
    print("Generating with regional injections...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "a natural landscape",
        num_steps=30,
        cfg_weight=7.5,
        seed=42
    ):
        latents = step_latents
    
    save_image(latents[0], "test3_regional_injection.png", sd)
    print("✓ Regional injection test complete")
    return latents, sd

def test_attention_manipulation():
    """Test 4: Attention Manipulation"""
    print("\n=== TEST 4: ATTENTION MANIPULATION ===")
    print("Testing cross and self-attention scaling...")
    
    sd = StableDiffusion.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    wrapper = CorePulseEnhanced(sd)
    
    # Configure attention manipulation
    wrapper.set_attention_manipulation(
        cross_scale=1.5,  # Amplify cross-attention (text influence)
        self_scale=0.8,   # Reduce self-attention (spatial coherence)
        layers=[4, 8, 12] # Target specific layers
    )
    
    # Add injections with attention control
    wrapper.add_injection(
        prompt="cyberpunk neon city",
        weight=0.6,
        cross_attention_scale=2.0,  # Strong text influence
        self_attention_scale=0.5    # Less spatial coherence
    )
    
    # Generate
    print("Generating with attention manipulation...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "futuristic urban environment",
        num_steps=30,
        cfg_weight=7.5,
        seed=42,
        enable_attention_control=True
    ):
        latents = step_latents
    
    save_image(latents[0], "test4_attention_manipulation.png", sd)
    print("✓ Attention manipulation test complete")
    return latents, sd

def test_multi_scale_control():
    """Test 5: Multi-Scale Control"""
    print("\n=== TEST 5: MULTI-SCALE CONTROL ===")
    print("Testing coarse-to-fine generation control...")
    
    sd = StableDiffusion.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    wrapper = CorePulseEnhanced(sd)
    
    # Configure multi-scale generation
    wrapper.set_multi_scale_control(
        scales=[0.25, 0.5, 1.0],  # Different resolution levels
        weights=[0.3, 0.3, 0.4]   # Contribution of each scale
    )
    
    # Add scale-aware injections
    wrapper.add_injection(
        prompt="abstract geometric patterns",
        weight=0.5,
        scale_levels=[0.25],  # Only at coarse scale
        scale_weights=[1.0]
    )
    
    wrapper.add_injection(
        prompt="intricate detailed ornaments",
        weight=0.5,
        scale_levels=[1.0],  # Only at fine scale
        scale_weights=[1.0]
    )
    
    # Generate
    print("Generating with multi-scale control...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "artistic decorative design",
        num_steps=30,
        cfg_weight=7.5,
        seed=42,
        enable_multi_scale=True
    ):
        latents = step_latents
    
    save_image(latents[0], "test5_multi_scale_control.png", sd)
    print("✓ Multi-scale control test complete")
    return latents, sd

def test_combined_features():
    """Test 6: All features combined"""
    print("\n=== TEST 6: COMBINED FEATURES ===")
    print("Testing all CorePulse features together...")
    
    sd = StableDiffusion.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    wrapper = CorePulseEnhanced(sd)
    
    # 1. Time-windowed prompt injection
    wrapper.add_injection(
        prompt="epic fantasy atmosphere",
        start_frac=0.0,
        end_frac=0.4,
        weight=0.6
    )
    
    # 2. Token-masked injection
    wrapper.add_token_masked_injection(
        prompt="majestic dragon breathing fire",
        focus_tokens="dragon fire",
        weight=0.7
    )
    
    # 3. Regional injection - castle on left
    wrapper.add_regional_injection(
        prompt="ancient stone castle on cliff",
        region=(0, 0, 256, 512),
        weight=0.8
    )
    
    # 4. Regional injection - dragon on right
    wrapper.add_regional_injection(
        prompt="massive red dragon in flight",
        region=(256, 0, 512, 512),
        weight=0.8
    )
    
    # 5. Attention manipulation
    wrapper.set_attention_manipulation(
        cross_scale=1.3,
        self_scale=0.9
    )
    
    # 6. Multi-scale control
    wrapper.set_multi_scale_control(
        scales=[0.25, 0.5, 1.0],
        weights=[0.2, 0.3, 0.5]
    )
    
    # Generate with all features
    print("Generating with all features enabled...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "fantasy battle scene",
        negative_text="blurry, low quality, abstract",
        num_steps=40,
        cfg_weight=8.0,
        seed=42,
        enable_attention_control=True,
        enable_multi_scale=True
    ):
        latents = step_latents
    
    save_image(latents[0], "test6_combined_features.png", sd)
    print("✓ Combined features test complete")
    return latents, sd

def main():
    """Run all CorePulse feature tests"""
    print("=" * 60)
    print("COREPULSE FEATURE TESTING SUITE")
    print("Testing all advanced features with visual validation")
    print("=" * 60)
    
    try:
        # Run individual feature tests
        test_prompt_injection()
        test_token_masking()
        test_regional_injection()
        test_attention_manipulation()
        test_multi_scale_control()
        test_combined_features()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("Check the generated images to validate features:")
        print("  - test1_prompt_injection.png")
        print("  - test2_token_masking.png")
        print("  - test3_regional_injection.png")
        print("  - test4_attention_manipulation.png")
        print("  - test5_multi_scale_control.png")
        print("  - test6_combined_features.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())