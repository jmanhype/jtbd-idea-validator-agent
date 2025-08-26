#!/usr/bin/env python3
"""
Quick test to verify hooks are actually being called.
"""

import sys
from pathlib import Path

# Add MLX examples to path
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples"))
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples/stable_diffusion"))

from stable_diffusion import StableDiffusion
from stable_diffusion.attn_hooks import (
    enable_hooks, disable_hooks, register_processor, 
    ATTN_HOOKS_ENABLED
)
import mlx.core as mx

class VerboseProcessor:
    """Processor that prints when called."""
    def __init__(self):
        self.call_count = 0
        
    def __call__(self, *, out=None, meta=None):
        self.call_count += 1
        if self.call_count <= 10:  # Only print first 10 calls
            print(f"✓ Hook called: block={meta.get('block_id')}, step={meta.get('step_idx')}")
        return out  # Return unchanged

# Initialize
print("Testing hook activation...")
print(f"Initial ATTN_HOOKS_ENABLED: {ATTN_HOOKS_ENABLED}")

# Enable hooks
enable_hooks()
from stable_diffusion.attn_hooks import ATTN_HOOKS_ENABLED
print(f"After enable_hooks(): {ATTN_HOOKS_ENABLED}")

# Register processor
processor = VerboseProcessor()
for block in ["down_0", "mid", "up_0"]:
    register_processor(block, processor)

# Load model and generate
sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)

print("\nGenerating with hooks enabled...")
latents = sd.generate_latents(
    "test",
    n_images=1,
    cfg_weight=7.5,
    num_steps=2,
    seed=42
)

for x in latents:
    mx.eval(x)

print(f"\nTotal hook calls: {processor.call_count}")

if processor.call_count > 0:
    print("✅ Hooks are working!")
else:
    print("❌ Hooks were not called!")