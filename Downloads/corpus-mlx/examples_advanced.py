#!/usr/bin/env python3
"""
Advanced CorePulse examples demonstrating attention manipulation,
per-block control, and multi-scale generation.
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

def save_image(latents, sd, filename):
    """Helper to decode and save image."""
    images = sd.autoencoder.decode(latents)
    images = mx.clip(images / 2 + 0.5, 0, 1)
    images = (images * 255).astype(mx.uint8)
    
    images_np = np.array(images)
    if images_np.ndim == 4:
        images_np = images_np[0]
    if images_np.shape[0] in [3, 4]:
        images_np = np.transpose(images_np, (1, 2, 0))
    
    img = Image.fromarray(images_np)
    img.save(filename)
    print(f"✓ Saved: {filename}")


def example_attention_manipulation():
    """
    Demonstrate fine-grained attention control across UNet blocks.
    """
    print("\n=== Example 1: Attention Manipulation ===")
    
    # Initialize
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    wrapper = CorePulseStableDiffusion(sd)
    
    # Create attention controller
    attn_ctrl = AttentionController(sd)
    
    # Boost attention in early blocks (structure)
    attn_ctrl.set_attention_scale("down_0", 1.5)
    attn_ctrl.set_attention_scale("down_1", 1.3)
    
    # Normal mid block
    attn_ctrl.set_attention_scale("mid", 1.0)
    
    # Reduce attention in late blocks (details)
    attn_ctrl.set_attention_scale("up_2", 0.8)
    attn_ctrl.set_attention_scale("up_3", 0.7)
    
    # Generate with controlled attention
    print("Generating with attention control...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "majestic eagle soaring through clouds",
        negative_text="blurry, low quality",
        num_steps=30,
        cfg_weight=7.5,
        seed=100
    ):
        latents = step_latents
    
    save_image(latents, sd, "example_attention_control.png")


def example_per_block_injection():
    """
    Demonstrate different prompts affecting different UNet blocks.
    """
    print("\n=== Example 2: Per-Block Injection ===")
    
    # Initialize
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    wrapper = CorePulseStableDiffusion(sd)
    
    # Create per-block controller
    block_ctrl = PerBlockInjectionController(sd)
    
    # Early blocks: Overall composition
    block_ctrl.add_block_injection(
        prompt="epic mountain landscape",
        blocks=["down_0", "down_1"],
        weight=0.8
    )
    
    # Mid blocks: Main subject
    block_ctrl.add_block_injection(
        prompt="ancient temple ruins",
        blocks=["mid", "up_0"],
        weight=0.7
    )
    
    # Late blocks: Fine details and atmosphere
    block_ctrl.add_block_injection(
        prompt="golden sunset lighting, mist and fog",
        blocks=["up_2", "up_3"],
        weight=0.6
    )
    
    # Prepare embeddings
    block_ctrl.prepare_embeddings()
    
    # Generate
    print("Generating with per-block control...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "mystical scene",
        num_steps=30,
        cfg_weight=8.0,
        seed=200
    ):
        latents = step_latents
    
    save_image(latents, sd, "example_per_block.png")


def example_attention_redistribution():
    """
    Demonstrate redistributing attention between tokens.
    """
    print("\n=== Example 3: Attention Redistribution ===")
    
    # Initialize
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    wrapper = CorePulseStableDiffusion(sd)
    
    # Create attention controller
    attn_ctrl = AttentionController(sd)
    
    # Example: Redistribute attention from background to subject
    # Token indices would need to be computed from actual tokenization
    background_tokens = [5, 6, 7]  # "landscape background scenery"
    subject_tokens = [2, 3]  # "dragon knight"
    
    attn_ctrl.redistribute_attention(
        from_tokens=background_tokens,
        to_tokens=subject_tokens,
        redistribution_weight=0.4,  # Move 40% attention to subject
        blocks=["mid", "up_0", "up_1"]  # Apply in mid-late blocks
    )
    
    # Generate
    print("Generating with attention redistribution...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "dragon knight in landscape background scenery",
        num_steps=30,
        cfg_weight=7.5,
        seed=300
    ):
        latents = step_latents
    
    save_image(latents, sd, "example_redistribution.png")


def example_multi_scale_control():
    """
    Demonstrate multi-scale generation with resolution-aware prompts.
    """
    print("\n=== Example 4: Multi-Scale Control ===")
    
    # Initialize
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    wrapper = CorePulseStableDiffusion(sd)
    
    # Create multi-scale controller
    scale_ctrl = MultiScaleController(sd)
    
    # Coarse scale: Overall structure (early steps, low resolution)
    scale_ctrl.add_scale_config(
        prompt="geometric abstract composition",
        resolution_scale=0.25,  # Low resolution
        start_frac=0.0,
        end_frac=0.3,
        weight=0.8
    )
    
    # Medium scale: Main elements (mid steps)
    scale_ctrl.add_scale_config(
        prompt="crystalline structures and patterns",
        resolution_scale=0.5,  # Medium resolution
        start_frac=0.2,
        end_frac=0.6,
        weight=0.7
    )
    
    # Fine scale: Details and textures (late steps, full resolution)
    scale_ctrl.add_scale_config(
        prompt="intricate fractal details, sharp edges",
        resolution_scale=1.0,  # Full resolution
        start_frac=0.5,
        end_frac=1.0,
        weight=0.6
    )
    
    # Generate
    print("Generating with multi-scale control...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "complex geometric art",
        num_steps=40,
        cfg_weight=8.0,
        seed=400
    ):
        latents = step_latents
    
    save_image(latents, sd, "example_multiscale.png")


def example_combined_advanced():
    """
    Combine all advanced features for maximum control.
    """
    print("\n=== Example 5: Combined Advanced Features ===")
    
    # Initialize
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    wrapper = CorePulseStableDiffusion(sd)
    
    # 1. Standard time-windowed injection
    wrapper.add_injection(
        prompt="ethereal dreamlike atmosphere",
        start_frac=0.0,
        end_frac=0.4,
        weight=0.6
    )
    
    # 2. Regional control with token masking
    wrapper.add_injection(
        prompt="glowing magical portal gateway",
        token_mask="portal gateway",
        region=("circle_pix", 256, 256, 100, 30),
        start_frac=0.3,
        end_frac=0.8,
        weight=0.7
    )
    
    # 3. Attention manipulation
    attn_ctrl = AttentionController(sd)
    attn_ctrl.set_attention_scale("down_*", 1.2)  # Boost early blocks
    attn_ctrl.set_attention_scale("up_*", 0.9)   # Reduce late blocks
    
    # 4. Per-block injection
    block_ctrl = PerBlockInjectionController(sd)
    block_ctrl.add_block_injection(
        prompt="cosmic nebula background",
        blocks=["down_0", "down_1"],
        weight=0.5
    )
    
    # Generate
    print("Generating with ALL advanced features...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "interdimensional scene",
        negative_text="boring, simple, plain",
        num_steps=50,
        cfg_weight=8.5,
        seed=500
    ):
        latents = step_latents
    
    save_image(latents, sd, "example_combined_advanced.png")


def example_dynamic_attention():
    """
    Demonstrate dynamic attention modification during generation.
    """
    print("\n=== Example 6: Dynamic Attention ===")
    
    # Initialize
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    wrapper = CorePulseStableDiffusion(sd)
    
    # Add a hook that modifies attention based on progress
    def dynamic_attention_hook(i, progress, x_t):
        # Early: Focus on structure
        if progress < 0.3:
            # Would modify attention weights here
            pass
        # Mid: Balance
        elif progress < 0.7:
            pass
        # Late: Focus on details
        else:
            pass
        return x_t
    
    wrapper.add_pre_step_hook(dynamic_attention_hook)
    
    # Generate
    print("Generating with dynamic attention...")
    latents = None
    for step_latents in wrapper.generate_latents(
        "futuristic cityscape with flying vehicles",
        num_steps=30,
        cfg_weight=7.5,
        seed=600
    ):
        latents = step_latents
    
    save_image(latents, sd, "example_dynamic_attention.png")


def run_all_examples():
    """Run all advanced examples."""
    print("=" * 60)
    print("COREPULSE ADVANCED EXAMPLES")
    print("=" * 60)
    
    # Run each example
    example_attention_manipulation()
    example_per_block_injection()
    example_attention_redistribution()
    example_multi_scale_control()
    example_combined_advanced()
    example_dynamic_attention()
    
    print("\n" + "=" * 60)
    print("✓ ALL ADVANCED EXAMPLES COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CorePulse Advanced Examples")
    parser.add_argument("--example", type=int, help="Run specific example (1-6)")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    
    args = parser.parse_args()
    
    if args.all:
        run_all_examples()
    elif args.example:
        examples = [
            example_attention_manipulation,
            example_per_block_injection,
            example_attention_redistribution,
            example_multi_scale_control,
            example_combined_advanced,
            example_dynamic_attention
        ]
        if 1 <= args.example <= len(examples):
            examples[args.example - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        # Default: run example 1
        example_attention_manipulation()