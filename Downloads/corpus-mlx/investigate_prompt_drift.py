#!/usr/bin/env python3
"""
Investigate prompt drift - Why is "sports car" generating living rooms?
Test different seeds, prompts, and parameters to isolate the issue.
"""

import sys
from pathlib import Path
import mlx.core as mx
from PIL import Image
import numpy as np

# Add MLX examples to path
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples"))
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples/stable_diffusion"))

from stable_diffusion import StableDiffusion


def generate_clean_baseline(prompt: str, seed: int, steps: int = 15):
    """Generate with no CorePulse interference - pure SD."""
    print(f"Testing: '{prompt}' with seed {seed}")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    mx.random.seed(seed)
    
    latents = sd.generate_latents(
        prompt,
        n_images=1,
        cfg_weight=7.5,
        num_steps=steps,
        seed=seed
    )
    
    x_t = None
    for x in latents:
        x_t = x
        mx.eval(x_t)
    
    image = sd.decode(x_t)
    mx.eval(image)
    
    img_array = (image[0] * 255).astype(mx.uint8)
    pil_image = Image.fromarray(np.array(img_array))
    
    mx.clear_cache()
    return pil_image


def investigation_suite():
    """Run comprehensive investigation."""
    
    print("\n" + "🔍"*50)
    print("   INVESTIGATING PROMPT DRIFT")
    print("🔍"*50)
    
    # Test 1: Same problematic prompt with different seeds
    print("\n1. SEED VARIATION TEST")
    print("="*40)
    
    problem_prompt = "a futuristic sports car"
    problem_seed = 5200  # The seed that gave us living room
    
    seeds_to_test = [problem_seed, problem_seed + 1, problem_seed + 50, 1000, 2000, 3000]
    
    for i, seed in enumerate(seeds_to_test):
        img = generate_clean_baseline(problem_prompt, seed)
        filename = f"investigate_seed_{seed}.png"
        img.save(filename)
        print(f"  Seed {seed}: Saved {filename}")
    
    # Test 2: Prompt variations
    print("\n2. PROMPT VARIATION TEST")
    print("="*40)
    
    car_prompts = [
        "a sports car",
        "a red sports car", 
        "a futuristic car",
        "a sports car in garage",
        "sports car, automotive, vehicle",
        "a sleek futuristic sports car, automotive photography"
    ]
    
    test_seed = 1000
    for i, prompt in enumerate(car_prompts):
        img = generate_clean_baseline(prompt, test_seed)
        filename = f"investigate_prompt_{i:02d}.png"
        img.save(filename)
        print(f"  '{prompt}': Saved {filename}")
    
    # Test 3: Different step counts
    print("\n3. STEP COUNT TEST")
    print("="*40)
    
    base_prompt = "a red sports car"
    base_seed = 1000
    step_counts = [10, 15, 20, 25]
    
    for steps in step_counts:
        img = generate_clean_baseline(base_prompt, base_seed, steps)
        filename = f"investigate_steps_{steps}.png"
        img.save(filename)
        print(f"  {steps} steps: Saved {filename}")
    
    # Test 4: Control prompts (should work reliably)  
    print("\n4. CONTROL PROMPTS TEST")
    print("="*40)
    
    control_prompts = [
        "a cat sitting on a chair",
        "a mountain landscape",
        "a portrait of a woman",
        "a house with red roof"
    ]
    
    for i, prompt in enumerate(control_prompts):
        img = generate_clean_baseline(prompt, 1000 + i)
        filename = f"investigate_control_{i:02d}.png"
        img.save(filename)
        print(f"  '{prompt}': Saved {filename}")
    
    print(f"\n{'='*50}")
    print("🔍 INVESTIGATION COMPLETE!")
    print(f"{'='*50}")
    print("Check the generated files to see:")
    print("  • investigate_seed_*.png - Same prompt, different seeds")
    print("  • investigate_prompt_*.png - Different car prompts")
    print("  • investigate_steps_*.png - Different step counts")
    print("  • investigate_control_*.png - Control prompts")
    print("\nThis will reveal if the issue is:")
    print("  - Seed-specific (only some seeds fail)")
    print("  - Prompt-specific (model doesn't understand 'futuristic sports car')")
    print("  - Parameter-specific (steps/cfg causing drift)")
    print("  - Model-specific (SD 2.1 base has issues with cars)")


if __name__ == "__main__":
    investigation_suite()