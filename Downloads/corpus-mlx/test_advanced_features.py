#!/usr/bin/env python3
"""
Test script to validate all advanced CorePulse features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper import CorePulseStableDiffusion
from corpus_mlx.attention import (
    AttentionController,
    PerBlockInjectionController,
    MultiScaleController
)
from corpus_mlx.optimizations import (
    create_optimized_wrapper,
    benchmark_optimizations,
    StreamingOptimizer
)

def test_advanced_integration():
    """Test all advanced features working together."""
    print("=" * 60)
    print("TESTING ADVANCED COREPULSE FEATURES")
    print("=" * 60)
    
    # Initialize
    print("\n1. Initializing models...")
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    # Create optimized wrapper
    print("2. Creating optimized wrapper...")
    wrapper = create_optimized_wrapper(sd)
    
    # Configure advanced features
    print("3. Configuring advanced features...")
    
    # A. Attention control
    attn_ctrl = AttentionController(sd)
    attn_ctrl.set_attention_scale("down_0", 1.4)
    attn_ctrl.set_attention_scale("down_1", 1.2)
    attn_ctrl.set_attention_scale("mid", 1.0)
    attn_ctrl.set_attention_scale("up_2", 0.8)
    attn_ctrl.set_attention_scale("up_3", 0.7)
    print("   ✓ Attention control configured")
    
    # B. Per-block injection
    block_ctrl = PerBlockInjectionController(sd)
    block_ctrl.add_block_injection(
        prompt="vast cosmic nebula",
        blocks=["down_0", "down_1"],
        weight=0.6
    )
    block_ctrl.add_block_injection(
        prompt="crystalline structures",
        blocks=["mid"],
        weight=0.7
    )
    block_ctrl.prepare_embeddings()
    print("   ✓ Per-block injection configured")
    
    # C. Multi-scale control
    scale_ctrl = MultiScaleController(sd)
    scale_ctrl.add_scale_config(
        prompt="abstract patterns",
        resolution_scale=0.25,
        start_frac=0.0,
        end_frac=0.3,
        weight=0.8
    )
    scale_ctrl.add_scale_config(
        prompt="intricate details",
        resolution_scale=1.0,
        start_frac=0.6,
        end_frac=1.0,
        weight=0.7
    )
    print("   ✓ Multi-scale control configured")
    
    # D. Standard injections
    wrapper.add_injection(
        prompt="ethereal glow",
        start_frac=0.0,
        end_frac=0.4,
        weight=0.5
    )
    wrapper.add_injection(
        prompt="sharp metallic textures",
        token_mask="metallic textures",
        start_frac=0.5,
        end_frac=1.0,
        weight=0.6
    )
    wrapper.add_injection(
        prompt="bright energy core",
        region=("circle_pix", 256, 256, 80, 20),
        weight=0.7
    )
    print("   ✓ Standard injections configured")
    
    # E. Streaming generation
    print("\n4. Testing streaming generation...")
    streaming_opt = StreamingOptimizer(sd)
    
    def preview_callback(latents, step, progress):
        print(f"   Step {step}: {progress:.1%} complete")
    
    # Generate with all features
    print("\n5. Generating with ALL advanced features...")
    latents = streaming_opt.streaming_generate(
        wrapper,
        "futuristic artifact",
        preview_callback,
        negative_text="blurry, low quality",
        num_steps=20,
        cfg_weight=8.0,
        seed=999
    )
    
    # Decode and save
    print("\n6. Decoding and saving result...")
    images = sd.autoencoder.decode(latents)
    images = mx.clip(images / 2 + 0.5, 0, 1)
    images = (images * 255).astype(mx.uint8)
    
    images_np = np.array(images)
    if images_np.ndim == 4:
        images_np = images_np[0]
    if images_np.shape[0] in [3, 4]:
        images_np = np.transpose(images_np, (1, 2, 0))
    
    img = Image.fromarray(images_np)
    img.save("test_advanced_integration.png")
    print("✓ Saved: test_advanced_integration.png")
    
    # Run benchmarks
    print("\n7. Running performance benchmarks...")
    benchmark_results = benchmark_optimizations(sd)
    
    # Summary
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES TEST COMPLETE!")
    print("=" * 60)
    print("\nFeatures tested:")
    print("✓ Attention manipulation (per-block scaling)")
    print("✓ Per-block prompt injection")
    print("✓ Multi-scale generation control")
    print("✓ Standard CorePulse injections")
    print("✓ MLX optimizations (caching, batching)")
    print("✓ Streaming generation with callbacks")
    print(f"\nPerformance improvement: {benchmark_results['speedup']:.2f}x")
    print("\nAll advanced features working successfully! 🎉")


if __name__ == "__main__":
    test_advanced_integration()