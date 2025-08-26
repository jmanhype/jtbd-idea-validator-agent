#!/usr/bin/env python3
"""
CorePulse Proper Fix - Address SD 2.1-base prompt adherence issues.
Research shows the model needs higher CFG and better prompt engineering.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mlx.core as mx
import re

# Add MLX examples to path
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples"))
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples/stable_diffusion"))

from stable_diffusion import StableDiffusion
from stable_diffusion import attn_hooks
from stable_diffusion import sigma_hooks


def generate_with_proper_settings(sd, prompt: str, processor=None, seed: int = 42, 
                                cfg_weight: float = 12.0, steps: int = 20):
    """Generate with research-backed settings for SD 2.1-base."""
    mx.random.seed(seed)
    
    if processor:
        attn_hooks.enable_hooks()
        for block in ['down_1', 'mid', 'up_1']:
            attn_hooks.register_processor(block, processor)
    
    # Use higher CFG and more steps based on research
    latents = sd.generate_latents(
        prompt,
        n_images=1,
        cfg_weight=cfg_weight,  # Research shows 7-12 for prompt adherence
        num_steps=steps,        # More steps for better quality
        seed=seed
    )
    
    x_t = None
    for x in latents:
        x_t = x
        mx.eval(x_t)
    
    image = sd.decode(x_t)
    mx.eval(image)
    
    # Cleanup
    if processor:
        attn_hooks.attention_registry.clear()
        attn_hooks.disable_hooks()
    mx.clear_cache()
    
    img_array = (image[0] * 255).astype(mx.uint8)
    return Image.fromarray(np.array(img_array))


def create_research_backed_test():
    """Test with proper SD 2.1 settings and prompts."""
    
    print("\n" + "🔬"*50)
    print("   RESEARCH-BACKED COREPULSE FIX")
    print("🔬"*50)
    print("\nApplying research findings:")
    print("  • CFG Scale: 12.0 (instead of 7.5)")
    print("  • Steps: 20 (instead of 15)")  
    print("  • Detailed prompts with style keywords")
    print("  • Physical descriptors instead of abstract concepts")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    
    # Research-backed prompts: detailed, specific, with style keywords
    test_cases = [
        ("photo of a red Ferrari sports car, automotive photography, professional lighting, 8K", 3000),
        ("close-up portrait photo of a smiling woman with brown hair, professional headshot, studio lighting", 3001),
        ("landscape photograph of snow-capped mountains and blue lake, nature photography, golden hour", 3002),
        ("photograph of a tabby cat sitting on wooden chair, pet photography, soft natural lighting", 3003)
    ]
    
    # Simple processor for gentle enhancement
    class ProperProcessor:
        def __init__(self, strength=0.08):
            self.strength = strength
            self.activations = []
        
        def __call__(self, *, out=None, meta=None):
            if out is None:
                return None
            
            sigma = meta.get('sigma', 0.0) if meta else 0.0
            step = meta.get('step_idx', 0) if meta else 0
            
            # Very gentle enhancement
            if sigma > 10:      # Structure
                factor = 1.0 + self.strength * 0.5
            elif sigma > 5:     # Content
                factor = 1.0 + self.strength * 0.8
            else:              # Details
                factor = 1.0 + self.strength * 1.0
            
            self.activations.append({'step': step, 'sigma': sigma, 'factor': factor})
            return out * factor
    
    results = []
    
    for i, (prompt, seed) in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Testing: {prompt[:50]}...")
        
        # Test different CFG values
        cfg_tests = [
            ("Low CFG", 7.5),
            ("Research CFG", 12.0),
            ("High CFG", 15.0)
        ]
        
        comparison_images = []
        
        for cfg_name, cfg_value in cfg_tests:
            print(f"  {cfg_name} (CFG {cfg_value})...")
            
            # Normal generation
            normal_img = generate_with_proper_settings(sd, prompt, None, seed, cfg_value, 20)
            
            # Enhanced generation
            processor = ProperProcessor(strength=0.08)
            enhanced_img = generate_with_proper_settings(sd, prompt, processor, seed, cfg_value, 20)
            
            comparison_images.extend([
                (normal_img, f"{cfg_name} Normal"),
                (enhanced_img, f"{cfg_name} Enhanced")
            ])
            
            activations = len(processor.activations)
            avg_factor = np.mean([a['factor'] for a in processor.activations]) if processor.activations else 1.0
            print(f"    Enhanced: {activations} adjustments, avg ×{avg_factor:.3f}")
        
        # Create comparison grid for this prompt
        create_cfg_comparison_grid(comparison_images, prompt, f"proper_fix_{i:02d}.png")
        results.append((prompt, comparison_images))
    
    return results


def create_cfg_comparison_grid(images: List[Tuple[Image.Image, str]], 
                              prompt: str, filename: str):
    """Create grid showing CFG scale effects."""
    
    if not images:
        return
    
    img_width, img_height = images[0][0].size
    num_images = len(images)
    
    # Layout: 3 rows (CFG levels) x 2 cols (normal/enhanced)
    cols = 2
    rows = 3
    
    padding = 12
    text_height = 35
    title_height = 50
    
    grid_width = cols * img_width + (cols + 1) * padding
    grid_height = title_height + rows * (img_height + text_height) + (rows + 1) * padding
    
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font_title = font_label = ImageFont.load_default()
    
    # Title
    title = f"CFG Scale Fix: {prompt[:45]}..."
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((grid_width - title_width) // 2, padding), title, fill='black', font=font_title)
    
    # Images in grid
    y_start = title_height + padding
    
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            if idx < len(images):
                img, label = images[idx]
                
                x = padding + col * (img_width + padding)
                y = y_start + row * (img_height + text_height + padding)
                
                # Image
                grid.paste(img, (x, y))
                
                # Label
                color = 'blue' if 'Enhanced' in label else 'gray'
                draw.text((x + 5, y + img_height + 5), label, fill=color, font=font_label)
    
    grid.save(filename)
    print(f"✅ Saved: {filename}")


def main():
    """Run research-backed fix demonstration."""
    
    results = create_research_backed_test()
    
    print(f"\n{'='*70}")
    print("📊 RESEARCH-BACKED FIX RESULTS") 
    print(f"{'='*70}")
    print("Generated CFG scale comparison grids:")
    
    for i, (prompt, _) in enumerate(results):
        print(f"  • proper_fix_{i:02d}.png - {prompt[:40]}...")
    
    print(f"\n🔬 Research findings applied:")
    print("  ✓ Higher CFG scale (12.0) for better prompt adherence")
    print("  ✓ More denoising steps (20) for quality")
    print("  ✓ Detailed prompts with style keywords") 
    print("  ✓ Professional photography terms")
    print("  ✓ Physical descriptors instead of abstract concepts")
    
    print(f"\n🎯 This should fix the prompt ignoring issue!")
    print("Compare Low CFG vs Research CFG vs High CFG results")


if __name__ == "__main__":
    main()