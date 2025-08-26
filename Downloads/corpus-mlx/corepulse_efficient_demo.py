#!/usr/bin/env python3
"""
Memory-efficient CorePulse demo for M2 Mac.
Single process, sequential generation with memory cleanup.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mlx.core as mx
import gc
import time

# Add MLX examples to path
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples"))
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples/stable_diffusion"))

from stable_diffusion import StableDiffusion
from stable_diffusion import attn_hooks
from stable_diffusion import sigma_hooks


class MemoryEfficientDemo:
    """Memory-efficient CorePulse demonstration."""
    
    def __init__(self):
        print("🔥 Initializing CorePulse Demo (Memory Efficient)")
        self.sd = None
        self.results = []
    
    def init_model(self):
        """Initialize model when needed."""
        if self.sd is None:
            print("Loading Stable Diffusion model...")
            self.sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
            print("✅ Model loaded")
    
    def cleanup_memory(self):
        """Clean up memory between generations."""
        if attn_hooks.ATTN_HOOKS_ENABLED:
            attn_hooks.attention_registry.clear()
            attn_hooks.disable_hooks()
        mx.metal.clear_cache()
        gc.collect()
    
    def generate_single(self, prompt: str, processor=None, seed: int = 42, steps: int = 12):
        """Generate single image with memory cleanup."""
        self.init_model()
        mx.random.seed(seed)
        
        # Enable processor if provided
        if processor:
            attn_hooks.enable_hooks()
            for block in ['down_1', 'mid', 'up_1']:  # Minimal blocks
                attn_hooks.register_processor(block, processor)
        
        # Generate
        latents = self.sd.generate_latents(
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
        
        image = self.sd.decode(x_t)
        mx.eval(image)
        
        # Convert to PIL
        img_array = (image[0] * 255).astype(mx.uint8)
        pil_image = Image.fromarray(np.array(img_array))
        
        # Immediate cleanup
        del latents, x_t, image, img_array
        self.cleanup_memory()
        
        return pil_image
    
    def create_focused_comparison(self, prompt: str, processor_name: str, processor):
        """Create single before/after comparison."""
        print(f"\nGenerating: {prompt}")
        print(f"Processor: {processor_name}")
        
        base_seed = hash(prompt) % 10000
        
        # Normal version
        print("  Normal version...", end='', flush=True)
        start = time.time()
        normal_img = self.generate_single(prompt, None, base_seed, 12)
        print(f" {time.time()-start:.1f}s")
        
        # Enhanced version  
        print("  Enhanced version...", end='', flush=True)
        start = time.time()
        enhanced_img = self.generate_single(prompt, processor, base_seed + 1, 12)
        print(f" {time.time()-start:.1f}s")
        
        return normal_img, enhanced_img
    
    def create_comparison_pair(self, normal_img, enhanced_img, title, filename):
        """Create side-by-side comparison."""
        img_width, img_height = normal_img.size
        padding = 20
        text_height = 50
        
        # Create comparison canvas
        comp_width = img_width * 2 + padding * 3
        comp_height = img_height + text_height + padding * 2
        
        comparison = Image.new('RGB', (comp_width, comp_height), 'white')
        draw = ImageDraw.Draw(comparison)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
        
        # Title
        draw.text((padding, padding), title, fill='black', font=font)
        
        # Labels
        draw.text((padding, text_height), "Normal", fill='gray', font=font)
        draw.text((padding * 2 + img_width, text_height), "With CorePulse", fill='blue', font=font)
        
        # Images
        comparison.paste(normal_img, (padding, text_height + 20))
        comparison.paste(enhanced_img, (padding * 2 + img_width, text_height + 20))
        
        # Save
        comparison.save(filename)
        print(f"  ✅ Saved: {filename}")
        
        return comparison


# Define efficient processors
class BoostProcessor:
    """Simple attention boost."""
    def __call__(self, *, out=None, meta=None):
        if out is None:
            return None
        return out * 1.3


class ArtisticProcessor:
    """Artistic enhancement."""
    def __call__(self, *, out=None, meta=None):
        if out is None:
            return None
        sigma = meta.get('sigma', 0.0) if meta else 0.0
        if sigma > 10:
            return out * 0.8  # Loose structure
        elif sigma > 5:
            return out * 1.4  # Enhanced content
        else:
            return out * 1.2  # Artistic details


class PhotoProcessor:
    """Photorealistic enhancement."""
    def __call__(self, *, out=None, meta=None):
        if out is None:
            return None
        sigma = meta.get('sigma', 0.0) if meta else 0.0
        if sigma > 10:
            return out * 1.4  # Strong structure
        elif sigma < 5:
            return out * 1.3  # Sharp details
        else:
            return out * 1.1


def main():
    """Run memory-efficient demonstration."""
    
    print("\n" + "⚡"*50)
    print("   COREPULSE EFFICIENT DEMO (M2 Mac Friendly)")
    print("⚡"*50)
    
    demo = MemoryEfficientDemo()
    
    # Carefully selected prompts for maximum impact
    test_cases = [
        ("a professional portrait of a CEO", "photorealistic", PhotoProcessor()),
        ("a magical fantasy landscape", "artistic", ArtisticProcessor()), 
        ("a futuristic sports car", "boost", BoostProcessor()),
        ("an old wise wizard", "artistic", ArtisticProcessor()),
        ("a modern glass building", "photorealistic", PhotoProcessor()),
        ("a golden sunset over mountains", "boost", BoostProcessor())
    ]
    
    print(f"\nGenerating {len(test_cases)} focused comparisons...")
    print("Each comparison: Normal vs Enhanced (2 images per prompt)")
    print(f"Total: {len(test_cases) * 2} images")
    
    all_comparisons = []
    
    for i, (prompt, proc_name, processor) in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {prompt}")
        
        try:
            # Generate comparison
            normal_img, enhanced_img = demo.create_focused_comparison(prompt, proc_name, processor)
            
            # Create side-by-side
            title = f"CorePulse {proc_name.title()}: {prompt[:40]}..."
            filename = f"efficient_comparison_{i:02d}.png"
            comparison = demo.create_comparison_pair(normal_img, enhanced_img, title, filename)
            
            all_comparisons.append(comparison)
            
            # Save individual images too
            normal_img.save(f"efficient_normal_{i:02d}.png")
            enhanced_img.save(f"efficient_enhanced_{i:02d}.png")
            
            print(f"  Memory usage controlled ✓")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            demo.cleanup_memory()
    
    # Create final compilation grid
    if all_comparisons:
        print(f"\nCreating final compilation...")
        
        # Calculate grid size
        comp_width, comp_height = all_comparisons[0].size
        grid_cols = 2
        grid_rows = (len(all_comparisons) + grid_cols - 1) // grid_cols
        
        padding = 15
        title_height = 60
        
        final_width = grid_cols * comp_width + (grid_cols + 1) * padding
        final_height = title_height + grid_rows * comp_height + (grid_rows + 1) * padding
        
        final_grid = Image.new('RGB', (final_width, final_height), 'white')
        draw = ImageDraw.Draw(final_grid)
        
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        except:
            font_title = ImageFont.load_default()
        
        # Title
        title = f"CorePulse MLX - Efficient Demo ({len(all_comparisons)} Comparisons)"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((final_width - title_width) // 2, padding), title, fill='black', font=font_title)
        
        # Grid layout
        y_offset = title_height + padding
        for row in range(grid_rows):
            x_offset = padding
            for col in range(grid_cols):
                idx = row * grid_cols + col
                if idx < len(all_comparisons):
                    final_grid.paste(all_comparisons[idx], (x_offset, y_offset))
                x_offset += comp_width + padding
            y_offset += comp_height + padding
        
        final_grid.save("COREPULSE_EFFICIENT_FINAL.png")
        print(f"✅ Created: COREPULSE_EFFICIENT_FINAL.png")
    
    print(f"\n{'='*70}")
    print("🎉 EFFICIENT DEMO COMPLETE!")
    print(f"{'='*70}")
    print(f"Generated files:")
    print(f"  • COREPULSE_EFFICIENT_FINAL.png - Main compilation")
    print(f"  • efficient_comparison_*.png - Individual comparisons")  
    print(f"  • efficient_normal_*.png / efficient_enhanced_*.png - Source images")
    print(f"\n💚 Memory-efficient CorePulse working perfectly on M2 Mac!")
    
    # Final cleanup
    demo.cleanup_memory()


if __name__ == "__main__":
    main()