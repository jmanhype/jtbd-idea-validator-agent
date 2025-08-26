#!/usr/bin/env python3
"""
Rapid-fire CorePulse demos - Generate many quick comparisons.
Fast generation with fewer steps for maximum variety.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mlx.core as mx
import time

# Add MLX examples to path
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples"))
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples/stable_diffusion"))

from stable_diffusion import StableDiffusion
from stable_diffusion import attn_hooks
from stable_diffusion import sigma_hooks


class QuickProcessor:
    """Quick effect processor."""
    def __init__(self, name: str, early_mult: float, mid_mult: float, late_mult: float):
        self.name = name
        self.early_mult = early_mult
        self.mid_mult = mid_mult  
        self.late_mult = late_mult
        
    def __call__(self, *, out=None, meta=None):
        if out is None:
            return None
            
        sigma = meta.get('sigma', 0.0) if meta else 0.0
        block = meta.get('block_id', '') if meta else ''
        
        if sigma > 10:  # Early
            return out * self.early_mult
        elif sigma > 5:  # Mid
            return out * self.mid_mult
        else:  # Late
            return out * self.late_mult


def generate_image(sd, prompt: str, processor=None, num_steps: int = 10, seed: int = 42):
    """Quick image generation."""
    mx.random.seed(seed)
    
    if processor:
        attn_hooks.enable_hooks()
        for block in ['down_1', 'mid', 'up_1']:  # Reduced blocks for speed
            attn_hooks.register_processor(block, processor)
    
    latents = sd.generate_latents(
        prompt,
        n_images=1,
        cfg_weight=7.5,
        num_steps=num_steps,
        seed=seed
    )
    
    x_t = None
    for x in latents:
        x_t = x
        mx.eval(x_t)
    
    image = sd.decode(x_t)
    mx.eval(image)
    
    if processor:
        attn_hooks.attention_registry.clear()
        attn_hooks.disable_hooks()
    
    img_array = (image[0] * 255).astype(mx.uint8)
    return Image.fromarray(np.array(img_array))


def create_rapid_comparisons():
    """Generate rapid comparisons."""
    
    print("\n" + "⚡"*35)
    print("   RAPID-FIRE COREPULSE DEMO")  
    print("⚡"*35)
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    # Define quick effects
    effects = {
        'boost': QuickProcessor('boost', 1.3, 1.4, 1.2),
        'artistic': QuickProcessor('artistic', 0.8, 1.5, 1.3),
        'sharp': QuickProcessor('sharp', 1.4, 1.2, 1.4),
        'soft': QuickProcessor('soft', 0.9, 1.0, 0.8),
        'dramatic': QuickProcessor('dramatic', 1.2, 1.6, 1.3)
    }
    
    # Rapid prompts for variety
    rapid_prompts = [
        "a red sports car",
        "a golden retriever puppy", 
        "a coffee cup on table",
        "a mountain sunset",
        "a city at night",
        "a flower garden",
        "a vintage camera",
        "a sailing boat",
        "a forest path",
        "a cozy fireplace",
        "a steam locomotive",
        "a butterfly on flower",
        "a lighthouse beacon",
        "a medieval sword", 
        "a space rocket",
        "a violin on stage",
        "a desert oasis",
        "a rainbow after storm",
        "a castle tower",
        "a market street"
    ]
    
    print(f"\nGenerating {len(rapid_prompts)} prompts x {len(effects)+1} versions = {len(rapid_prompts)*(len(effects)+1)} total images")
    
    all_results = []
    
    for i, prompt in enumerate(rapid_prompts):
        print(f"\n{i+1:2d}/{len(rapid_prompts)}: {prompt}")
        
        result = {'prompt': prompt, 'images': []}
        base_seed = 2000 + i * 10
        
        # Normal version
        print(f"    Normal...", end='', flush=True)
        start = time.time()
        normal_img = generate_image(sd, prompt, None, 10, base_seed)
        normal_path = f"rapid_normal_{i:02d}.png"
        normal_img.save(normal_path)
        print(f" {time.time()-start:.1f}s")
        
        result['images'].append(('normal', normal_path, normal_img))
        
        # Enhanced versions
        for effect_name, processor in effects.items():
            print(f"    {effect_name.title()}...", end='', flush=True)
            start = time.time()
            enhanced_img = generate_image(sd, prompt, processor, 10, base_seed + 1)
            enhanced_path = f"rapid_{effect_name}_{i:02d}.png"
            enhanced_img.save(enhanced_path)
            print(f" {time.time()-start:.1f}s")
            
            result['images'].append((effect_name, enhanced_path, enhanced_img))
        
        all_results.append(result)
        
        # Progress update
        if (i + 1) % 5 == 0:
            print(f"\n  ✅ Completed {i+1}/{len(rapid_prompts)} prompts")
    
    return all_results


def create_rapid_grid(results, grid_name: str):
    """Create grid from rapid results."""
    
    print(f"\n{'='*70}")
    print("📊 CREATING RAPID COMPARISON GRID")
    print(f"{'='*70}")
    
    if not results:
        print("No results to grid!")
        return
    
    # Use subset for manageable grid size
    subset = results[:12]  # First 12 prompts
    effects_to_show = ['normal', 'boost', 'artistic', 'dramatic']  # 4 versions each
    
    print(f"Creating grid: {len(subset)} prompts x {len(effects_to_show)} effects")
    
    # Calculate grid dimensions
    img_width, img_height = results[0]['images'][0][2].size
    cols = len(effects_to_show)
    rows = len(subset)
    
    padding = 10
    label_height = 25
    title_height = 60
    
    grid_width = cols * img_width + (cols + 1) * padding
    grid_height = title_height + rows * (img_height + label_height + padding) + padding
    
    # Create canvas
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Main title
    title = f"CorePulse Rapid-Fire Demo ({len(subset)}x{len(effects_to_show)} = {len(subset)*len(effects_to_show)} images)"
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((grid_width - title_width) // 2, padding), title, fill='black', font=font_title)
    
    # Column headers
    y = title_height - 20
    for col, effect in enumerate(effects_to_show):
        x = padding + col * (img_width + padding) + img_width // 2 - 30
        draw.text((x, y), effect.title(), fill='blue' if effect != 'normal' else 'gray', font=font_label)
    
    # Draw grid
    y_offset = title_height
    
    for row, result in enumerate(subset):
        # Row label (prompt)
        prompt_short = result['prompt'][:20] + "..." if len(result['prompt']) > 20 else result['prompt']
        draw.text((5, y_offset + img_height // 2), prompt_short, fill='black', font=font_small)
        
        # Images in row
        for col, effect_name in enumerate(effects_to_show):
            # Find matching image
            img = None
            for name, path, image in result['images']:
                if name == effect_name:
                    img = image
                    break
            
            if img:
                x = padding + col * (img_width + padding)
                grid.paste(img, (x, y_offset))
        
        y_offset += img_height + label_height + padding
    
    # Save grid
    grid.save(grid_name)
    print(f"✅ Created: {grid_name}")


def create_side_by_side_samples(results):
    """Create focused side-by-side comparisons."""
    
    print(f"\n{'='*70}")
    print("📸 CREATING SIDE-BY-SIDE SAMPLES")
    print(f"{'='*70}")
    
    # Pick best examples
    sample_indices = [0, 3, 7, 11, 15]  # Spread across different prompts
    effects_to_compare = ['normal', 'boost', 'dramatic']
    
    for idx in sample_indices:
        if idx < len(results):
            result = results[idx]
            prompt = result['prompt']
            
            print(f"Creating sample: {prompt}")
            
            # Find images
            comparison_images = []
            for effect in effects_to_compare:
                for name, path, img in result['images']:
                    if name == effect:
                        comparison_images.append((effect, img))
                        break
            
            if len(comparison_images) == len(effects_to_compare):
                # Create side-by-side
                img_width, img_height = comparison_images[0][1].size
                padding = 20
                text_height = 40
                
                sample_width = len(comparison_images) * img_width + (len(comparison_images) + 1) * padding
                sample_height = img_height + text_height * 2 + padding * 2
                
                sample = Image.new('RGB', (sample_width, sample_height), 'white')
                draw = ImageDraw.Draw(sample)
                
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
                except:
                    font = ImageFont.load_default()
                
                # Title
                draw.text((padding, padding), f"'{prompt}'", fill='black', font=font)
                
                # Images with labels
                x_offset = padding
                for effect, img in comparison_images:
                    # Label
                    label_color = 'blue' if effect != 'normal' else 'gray'
                    draw.text((x_offset, text_height + padding), effect.title(), fill=label_color, font=font)
                    
                    # Image
                    sample.paste(img, (x_offset, text_height + padding + 20))
                    x_offset += img_width + padding
                
                # Save
                safe_prompt = prompt.replace(' ', '_').replace('/', '').replace('\\', '')[:20]
                sample_name = f"rapid_sample_{idx}_{safe_prompt}.png"
                sample.save(sample_name)
                print(f"  ✅ Saved: {sample_name}")


def main():
    """Run rapid-fire demonstration."""
    
    start_time = time.time()
    
    try:
        # Generate all comparisons
        results = create_rapid_comparisons()
        
        # Create visualizations
        create_rapid_grid(results, "COREPULSE_RAPID_GRID.png")
        create_side_by_side_samples(results)
        
        total_time = time.time() - start_time
        total_images = sum(len(r['images']) for r in results)
        
        print(f"\n{'='*70}")
        print("🎉 RAPID-FIRE DEMO COMPLETE!")
        print(f"{'='*70}")
        print(f"Generated {total_images} images in {total_time:.1f} seconds")
        print(f"Average: {total_time/total_images:.1f}s per image")
        print(f"\nFiles created:")
        print(f"  • COREPULSE_RAPID_GRID.png - Main comparison grid")
        print(f"  • rapid_sample_*.png - Focused comparisons")
        print(f"  • rapid_*.png - Individual images")
        
        print(f"\n⚡ CorePulse processes images FAST on Apple Silicon!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()