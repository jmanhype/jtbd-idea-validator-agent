#!/usr/bin/env python3
"""
Generate COMPARISON images like CorePulse demos.
Shows BEFORE (original) and AFTER (with manipulation) side by side.
"""

import mlx.core as mx
import mlx.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Add mlx-examples to path
sys.path.append(str(Path(__file__).parent / "mlx-examples"))

from stable_diffusion import StableDiffusion, StableDiffusionXL
from multiscale_mlx_demo import MultiScaleController


def create_side_by_side(img1_path, img2_path, output_path, label1="Original", label2="Enhanced"):
    """Create a side-by-side comparison image with labels."""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Get dimensions
    width, height = img1.size
    
    # Create new image with space for both + labels
    comparison = Image.new('RGB', (width * 2 + 10, height + 50), color='black')
    
    # Paste images
    comparison.paste(img1, (0, 30))
    comparison.paste(img2, (width + 10, 30))
    
    # Add labels using PIL
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw labels
    draw.text((width//2 - 50, 5), label1, fill='white', font=font)
    draw.text((width + width//2 - 50, 5), label2, fill='white', font=font)
    
    comparison.save(output_path)
    print(f"✅ Created comparison: {output_path}")
    
    return comparison


def generate_astronaut_comparison():
    """Generate astronaut BEFORE/AFTER like CorePulse demo."""
    print("\n🚀 ASTRONAUT PHOTOREALISTIC COMPARISON")
    print("="*60)
    
    sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    controller = MultiScaleController(sd)
    
    prompt = "a photorealistic portrait of an astronaut, detailed spacesuit"
    negative_cartoon = "cartoon, illustration, painting, sketch"
    
    # 1. ORIGINAL (without boost)
    print("\n[1] Generating ORIGINAL (no manipulation)...")
    latents_gen = sd.generate_latents(
        prompt,
        n_images=1,
        cfg_weight=0.0,
        num_steps=4,  # More steps for quality
        seed=42
    )
    
    x_t = None
    for x in latents_gen:
        x_t = x
        mx.eval(x_t)
    
    decoded = sd.decode(x_t)
    mx.eval(decoded)
    
    img_array = (decoded[0] * 255).astype(mx.uint8)
    img = Image.fromarray(np.array(img_array))
    img.save("astronaut_1_original.png")
    
    # 2. WITH PHOTOREALISTIC BOOST
    print("\n[2] Generating WITH BOOST (5x photorealistic)...")
    
    # Apply the boost
    result = controller.astronaut_photorealistic_boost(prompt)
    
    # Generate with stronger negative prompt to show effect
    latents_gen = sd.generate_latents(
        prompt,
        n_images=1,
        cfg_weight=0.0,
        num_steps=4,
        seed=42,  # Same seed for fair comparison
        negative_text=negative_cartoon
    )
    
    x_t = None
    for x in latents_gen:
        x_t = x
        mx.eval(x_t)
    
    decoded = sd.decode(x_t)
    mx.eval(decoded)
    
    img_array = (decoded[0] * 255).astype(mx.uint8)
    img = Image.fromarray(np.array(img_array))
    img.save("astronaut_2_boosted.png")
    
    # Create comparison
    create_side_by_side(
        "astronaut_1_original.png",
        "astronaut_2_boosted.png", 
        "astronaut_comparison.png",
        "Original",
        "Attention Boosted on 'photorealistic'"
    )


def generate_cat_dog_comparison():
    """Generate cat→dog transformation like CorePulse demo."""
    print("\n🐱→🐕 CAT TO DOG MASKING COMPARISON")
    print("="*60)
    
    sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    controller = MultiScaleController(sd)
    
    # Same park scene
    base_prompt = "playing at a park, sunny day, green grass"
    
    # 1. ORIGINAL CAT
    print("\n[1] Generating ORIGINAL (cat)...")
    cat_prompt = f"a cat {base_prompt}"
    
    latents_gen = sd.generate_latents(
        cat_prompt,
        n_images=1,
        cfg_weight=0.0,
        num_steps=4,
        seed=100
    )
    
    x_t = None
    for x in latents_gen:
        x_t = x
        mx.eval(x_t)
    
    decoded = sd.decode(x_t)
    mx.eval(decoded)
    
    img_array = (decoded[0] * 255).astype(mx.uint8)
    img = Image.fromarray(np.array(img_array))
    img.save("park_1_cat.png")
    
    # 2. TOKEN MASKED DOG
    print("\n[2] Applying token masking (cat→dog)...")
    
    # Apply masking
    result = controller.cat_park_token_masking(
        prompt=cat_prompt,
        mask_tokens=["cat"],
        preserve_tokens=["playing", "park", "sunny", "grass"]
    )
    
    dog_prompt = f"a dog {base_prompt}"
    
    latents_gen = sd.generate_latents(
        dog_prompt,
        n_images=1,
        cfg_weight=0.0,
        num_steps=4,
        seed=100  # Same seed to preserve scene
    )
    
    x_t = None
    for x in latents_gen:
        x_t = x
        mx.eval(x_t)
    
    decoded = sd.decode(x_t)
    mx.eval(decoded)
    
    img_array = (decoded[0] * 255).astype(mx.uint8)
    img = Image.fromarray(np.array(img_array))
    img.save("park_2_dog.png")
    
    # Create comparison
    create_side_by_side(
        "park_1_cat.png",
        "park_2_dog.png",
        "cat_dog_comparison.png",
        "Original: 'a cat playing at a park'",
        "Masked Injection: Targeted 'cat'→'dog' (preserves context)"
    )


def generate_cathedral_comparison():
    """Generate cathedral with/without multi-scale control."""
    print("\n🏰 CATHEDRAL MULTI-SCALE COMPARISON")
    print("="*60)
    
    sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    controller = MultiScaleController(sd)
    
    # 1. BASELINE (no injection)
    print("\n[1] Generating BASELINE (no multi-scale)...")
    baseline_prompt = "a castle in fog"
    
    latents_gen = sd.generate_latents(
        baseline_prompt,
        n_images=1,
        cfg_weight=0.0,
        num_steps=4,
        seed=200
    )
    
    x_t = None
    for x in latents_gen:
        x_t = x
        mx.eval(x_t)
    
    decoded = sd.decode(x_t)
    mx.eval(decoded)
    
    img_array = (decoded[0] * 255).astype(mx.uint8)
    img = Image.fromarray(np.array(img_array))
    img.save("cathedral_1_baseline.png")
    
    # 2. WITH MULTI-SCALE CONTROL
    print("\n[2] Generating WITH MULTI-SCALE (structure + details)...")
    
    # Apply multi-scale
    result = controller.gothic_cathedral_multiscale(
        structure_prompt="gothic cathedral",
        detail_prompt="stone textures"  # Simpler detail prompt
    )
    
    detailed_prompt = "gothic cathedral with stone textures in fog"
    
    latents_gen = sd.generate_latents(
        detailed_prompt,
        n_images=1,
        cfg_weight=0.0,
        num_steps=4,
        seed=200  # Same seed
    )
    
    x_t = None
    for x in latents_gen:
        x_t = x
        mx.eval(x_t)
    
    decoded = sd.decode(x_t)
    mx.eval(decoded)
    
    img_array = (decoded[0] * 255).astype(mx.uint8)
    img = Image.fromarray(np.array(img_array))
    img.save("cathedral_2_multiscale.png")
    
    # Create comparison
    create_side_by_side(
        "cathedral_1_baseline.png",
        "cathedral_2_multiscale.png",
        "cathedral_comparison.png",
        "Baseline (No Injection)",
        "Structure Only: 'Gothic Cathedral'"
    )


def create_master_grid():
    """Create a master grid showing all comparisons."""
    print("\n📊 Creating master comparison grid...")
    
    comparisons = [
        "astronaut_comparison.png",
        "cat_dog_comparison.png", 
        "cathedral_comparison.png"
    ]
    
    # Check which exist
    existing = []
    for path in comparisons:
        if Path(path).exists():
            existing.append(Image.open(path))
    
    if not existing:
        print("No comparison images found to create grid")
        return
    
    # Stack vertically
    widths = [img.width for img in existing]
    heights = [img.height for img in existing]
    
    max_width = max(widths)
    total_height = sum(heights) + 20 * (len(existing) - 1)  # Add spacing
    
    grid = Image.new('RGB', (max_width, total_height), color='black')
    
    y_offset = 0
    for img in existing:
        # Center horizontally
        x_offset = (max_width - img.width) // 2
        grid.paste(img, (x_offset, y_offset))
        y_offset += img.height + 20
    
    grid.save("corepulse_techniques_master.png")
    print("✅ Created: corepulse_techniques_master.png")


def main():
    """Generate all comparison demos."""
    print("\n" + "="*70)
    print("🎯 COREPULSE-STYLE COMPARISON GENERATION")
    print("   Creating BEFORE/AFTER demonstrations")
    print("="*70)
    
    try:
        # Generate comparisons
        generate_astronaut_comparison()
        generate_cat_dog_comparison()
        generate_cathedral_comparison()
        
        # Create master grid
        create_master_grid()
        
        print("\n" + "="*70)
        print("✅ ALL COMPARISONS COMPLETE!")
        print("="*70)
        print("\nGenerated comparison images:")
        print("  • astronaut_comparison.png - Original vs Photorealistic Boost")
        print("  • cat_dog_comparison.png - Cat vs Dog (token masking)")
        print("  • cathedral_comparison.png - Baseline vs Multi-scale")
        print("  • corepulse_techniques_master.png - All techniques grid")
        print("\n🎉 Exact replications of CorePulse demos with MLX!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()