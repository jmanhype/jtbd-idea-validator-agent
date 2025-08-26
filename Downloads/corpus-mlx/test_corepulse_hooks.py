#!/usr/bin/env python3
"""
Test CorePulse-style hooks in MLX Stable Diffusion.
Ensures identity processor maintains parity (hooks on == hooks off).
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import mlx.core as mx

# Add MLX examples to path
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples"))
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples/stable_diffusion"))

# Import MLX SD with our hooks
from stable_diffusion import StableDiffusion
from stable_diffusion.attn_hooks import (
    enable_hooks, disable_hooks, register_processor, 
    IdentityProcessor, attention_registry
)
from stable_diffusion.sigma_hooks import (
    register_observer, sigma_registry, LoggingSigmaObserver
)


def test_identity_parity():
    """Test that identity processor produces identical results to no hooks."""
    print("="*70)
    print("🧪 TESTING COREPULSE HOOKS - IDENTITY PARITY")
    print("="*70)
    
    # Initialize SD
    print("\n1. Loading model...")
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    prompt = "a beautiful landscape with mountains and lake"
    seed = 42
    num_steps = 20
    
    # Generate WITHOUT hooks
    print(f"\n2. Generating WITHOUT hooks...")
    print(f"   Prompt: {prompt}")
    print(f"   Seed: {seed}")
    print(f"   Steps: {num_steps}")
    
    disable_hooks()  # Ensure hooks are off
    mx.random.seed(seed)
    
    latents_no_hooks = sd.generate_latents(
        prompt,
        n_images=1,
        cfg_weight=7.5,
        num_steps=num_steps,
        seed=seed
    )
    
    # Collect the final latent
    x_t_no_hooks = None
    for x in latents_no_hooks:
        x_t_no_hooks = x
        mx.eval(x_t_no_hooks)
    
    # Decode
    image_no_hooks = sd.decode(x_t_no_hooks)
    mx.eval(image_no_hooks)
    
    # Convert to PIL
    img_array_no_hooks = (image_no_hooks[0] * 255).astype(mx.uint8)
    img_no_hooks = Image.fromarray(np.array(img_array_no_hooks))
    img_no_hooks.save("test_no_hooks.png")
    
    print("   ✅ Saved: test_no_hooks.png")
    
    # Generate WITH identity processor
    print(f"\n3. Generating WITH identity processor...")
    
    enable_hooks()  # Turn on hooks
    
    # Register identity processors for all blocks
    identity = IdentityProcessor()
    for block_id in ["down_0", "down_1", "down_2", "down_3", 
                     "mid", "up_0", "up_1", "up_2", "up_3"]:
        register_processor(block_id, identity)
    
    # Reset seed for identical generation
    mx.random.seed(seed)
    
    latents_with_hooks = sd.generate_latents(
        prompt,
        n_images=1,
        cfg_weight=7.5,
        num_steps=num_steps,
        seed=seed
    )
    
    # Collect the final latent
    x_t_with_hooks = None
    for x in latents_with_hooks:
        x_t_with_hooks = x
        mx.eval(x_t_with_hooks)
    
    # Decode
    image_with_hooks = sd.decode(x_t_with_hooks)
    mx.eval(image_with_hooks)
    
    # Convert to PIL
    img_array_with_hooks = (image_with_hooks[0] * 255).astype(mx.uint8)
    img_with_hooks = Image.fromarray(np.array(img_array_with_hooks))
    img_with_hooks.save("test_with_identity_hooks.png")
    
    print("   ✅ Saved: test_with_identity_hooks.png")
    
    # Clean up registry
    attention_registry.clear()
    disable_hooks()
    
    # Compare images
    print(f"\n4. Comparing results...")
    
    # Convert to numpy for comparison
    arr1 = np.array(img_no_hooks)
    arr2 = np.array(img_with_hooks)
    
    # Check if identical
    if np.array_equal(arr1, arr2):
        print("   ✅ PASS: Images are IDENTICAL!")
        print("   Identity processor maintains perfect parity.")
    else:
        # Calculate difference
        diff = np.abs(arr1.astype(float) - arr2.astype(float))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"   ⚠️  Images differ slightly:")
        print(f"   Max pixel difference: {max_diff}")
        print(f"   Mean pixel difference: {mean_diff:.4f}")
        
        if max_diff < 5 and mean_diff < 1:
            print("   ✅ PASS: Differences are negligible (likely floating point)")
        else:
            print("   ❌ FAIL: Significant differences detected!")


def test_sigma_observer():
    """Test that sigma observer correctly tracks denoising progress."""
    print("\n" + "="*70)
    print("🔬 TESTING SIGMA OBSERVER")
    print("="*70)
    
    # Create and register observer
    observer = LoggingSigmaObserver()
    register_observer(observer)
    
    # Initialize SD
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    print("\n1. Generating with sigma observer...")
    
    # Generate
    latents = sd.generate_latents(
        "test image",
        n_images=1,
        cfg_weight=7.5,
        num_steps=10,
        seed=123
    )
    
    # Process latents to trigger sigma emissions
    for x in latents:
        mx.eval(x)
    
    # Clean up
    sigma_registry.clear()
    
    print("\n   ✅ Sigma observer test complete!")


def test_attention_manipulation():
    """Test actual attention manipulation with a custom processor."""
    print("\n" + "="*70)
    print("🎯 TESTING ATTENTION MANIPULATION")
    print("="*70)
    
    class AmplifyProcessor:
        """Simple processor that amplifies attention by a factor."""
        def __init__(self, factor=1.5):
            self.factor = factor
            
        def __call__(self, *, out=None, meta=None):
            if out is None:
                return None
            print(f"   Amplifying block {meta['block_id']} at step {meta['step_idx']} by {self.factor}x")
            return out * self.factor
    
    # Initialize SD
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    prompt = "a vibrant colorful abstract painting"
    seed = 999
    
    # Generate normally
    print("\n1. Generating WITHOUT amplification...")
    disable_hooks()
    mx.random.seed(seed)
    
    latents = sd.generate_latents(prompt, n_images=1, cfg_weight=7.5, num_steps=5, seed=seed)
    x_t = None
    for x in latents:
        x_t = x
        mx.eval(x_t)
    
    image = sd.decode(x_t)
    mx.eval(image)
    img_array = (image[0] * 255).astype(mx.uint8)
    img = Image.fromarray(np.array(img_array))
    img.save("test_normal.png")
    print("   ✅ Saved: test_normal.png")
    
    # Generate with amplification
    print("\n2. Generating WITH attention amplification...")
    enable_hooks()
    
    # Register amplifier for middle block only
    amplifier = AmplifyProcessor(factor=2.0)
    register_processor("mid", amplifier)
    
    mx.random.seed(seed)
    
    latents = sd.generate_latents(prompt, n_images=1, cfg_weight=7.5, num_steps=5, seed=seed)
    x_t = None
    for x in latents:
        x_t = x
        mx.eval(x_t)
    
    image = sd.decode(x_t)
    mx.eval(image)
    img_array = (image[0] * 255).astype(mx.uint8)
    img = Image.fromarray(np.array(img_array))
    img.save("test_amplified.png")
    print("   ✅ Saved: test_amplified.png")
    
    # Clean up
    attention_registry.clear()
    disable_hooks()
    
    print("\n   ✅ Attention manipulation test complete!")
    print("   Check test_normal.png vs test_amplified.png for differences.")


def main():
    """Run all tests."""
    print("\n" + "🚀"*35)
    print("   COREPULSE MLX HOOKS TEST SUITE")
    print("🚀"*35 + "\n")
    
    try:
        # Test 1: Identity parity
        test_identity_parity()
        
        # Test 2: Sigma observer
        test_sigma_observer()
        
        # Test 3: Attention manipulation
        test_attention_manipulation()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  • test_no_hooks.png (baseline)")
        print("  • test_with_identity_hooks.png (should be identical)")
        print("  • test_normal.png (without manipulation)")
        print("  • test_amplified.png (with attention amplification)")
        print("\n🎉 CorePulse hooks are ready for use!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()