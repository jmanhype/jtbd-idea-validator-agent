#!/usr/bin/env python3
"""
Stabilized CorePulse - Fix oscillations with controlled attention manipulation.
Preserves semantic consistency while providing targeted enhancements.
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


class StabilizedProcessor:
    """
    Stabilized processor that preserves semantic consistency.
    Uses gentle, progressive enhancements instead of dramatic shifts.
    """
    
    def __init__(self, enhancement_type: str, strength: float = 0.15):
        self.enhancement_type = enhancement_type
        self.strength = strength  # Much smaller multipliers
        self.activations = []
        
        # Define enhancement strategies with gentle multipliers
        self.strategies = {
            'photorealistic': {
                'early_structure': 1.0 + strength * 0.5,    # Very gentle structure
                'mid_content': 1.0 + strength * 0.3,        # Subtle content
                'late_detail': 1.0 + strength * 0.8,        # Moderate detail enhancement
            },
            'artistic': {
                'early_structure': 1.0 - strength * 0.3,    # Slightly softer structure
                'mid_content': 1.0 + strength * 0.6,        # Enhanced artistic content
                'late_detail': 1.0 + strength * 0.4,        # Artistic details
            },
            'sharp': {
                'early_structure': 1.0 + strength * 0.4,    # Crisper structure
                'mid_content': 1.0 + strength * 0.2,        # Maintain content
                'late_detail': 1.0 + strength * 1.0,        # Sharp details
            },
            'soft': {
                'early_structure': 1.0 - strength * 0.2,    # Softer structure
                'mid_content': 1.0,                          # Preserve content
                'late_detail': 1.0 - strength * 0.4,        # Softer details
            },
            'vibrant': {
                'early_structure': 1.0,                      # Preserve structure
                'mid_content': 1.0 + strength * 0.7,        # Enhanced colors
                'late_detail': 1.0 + strength * 0.3,        # Vibrant details
            }
        }
        
        self.strategy = self.strategies.get(enhancement_type, self.strategies['photorealistic'])
    
    def __call__(self, *, out=None, meta: Dict[str, Any]) -> Optional[Any]:
        if out is None:
            return None
            
        sigma = meta.get('sigma', 0.0)
        block = meta.get('block_id', '')
        step = meta.get('step_idx', 0)
        
        # Determine enhancement factor based on denoising stage
        if sigma > 10:  # Early - structure phase
            factor = self.strategy['early_structure']
        elif sigma > 5:  # Mid - content phase  
            factor = self.strategy['mid_content']
        else:  # Late - detail phase
            factor = self.strategy['late_detail']
        
        # Apply block-specific modulation (very gentle)
        if 'down' in block and sigma > 10:
            factor *= 1.0 + self.strength * 0.2  # Slight structure boost
        elif 'mid' in block and 5 < sigma <= 10:
            factor *= 1.0 + self.strength * 0.3  # Moderate content boost
        elif 'up' in block and sigma <= 5:
            factor *= 1.0 + self.strength * 0.4  # Detail boost
        
        # Log activation
        self.activations.append({
            'step': step,
            'block': block, 
            'sigma': float(sigma),
            'factor': float(factor)
        })
        
        return out * factor


class SemanticPreservationProcessor:
    """
    Processor that preserves semantic content while enhancing quality.
    Uses prompt-aware attention modulation.
    """
    
    def __init__(self, prompt: str, enhancement_mode: str = "quality"):
        self.prompt = prompt.lower()
        self.enhancement_mode = enhancement_mode
        self.activations = []
        
        # Analyze prompt for content preservation
        self.is_portrait = any(word in self.prompt for word in ['portrait', 'person', 'face', 'man', 'woman', 'people'])
        self.is_landscape = any(word in self.prompt for word in ['landscape', 'scenery', 'mountain', 'forest', 'nature'])
        self.is_object = any(word in self.prompt for word in ['car', 'building', 'object', 'product'])
        self.is_artistic = any(word in self.prompt for word in ['art', 'painting', 'artistic', 'style'])
        
    def __call__(self, *, out=None, meta: Dict[str, Any]) -> Optional[Any]:
        if out is None:
            return None
            
        sigma = meta.get('sigma', 0.0)
        block = meta.get('block_id', '')
        step = meta.get('step_idx', 0)
        
        # Semantic preservation strategy
        base_strength = 0.1  # Very conservative
        
        if self.is_portrait:
            # For portraits, focus on detail enhancement without changing identity
            if sigma > 10:  # Early - preserve basic structure
                factor = 1.0 + base_strength * 0.3
            elif sigma > 5:  # Mid - preserve facial features
                factor = 1.0 + base_strength * 0.2  
            else:  # Late - enhance skin/hair details
                factor = 1.0 + base_strength * 0.8
                
        elif self.is_landscape:
            # For landscapes, enhance depth and atmosphere
            if sigma > 10:  # Early - gentle composition
                factor = 1.0 + base_strength * 0.4
            elif sigma > 5:  # Mid - atmosphere enhancement
                factor = 1.0 + base_strength * 0.6
            else:  # Late - texture details
                factor = 1.0 + base_strength * 0.5
                
        elif self.is_object:
            # For objects, enhance form and materials
            if sigma > 10:  # Early - preserve object shape
                factor = 1.0 + base_strength * 0.5
            elif sigma > 5:  # Mid - material properties
                factor = 1.0 + base_strength * 0.4
            else:  # Late - surface details
                factor = 1.0 + base_strength * 0.7
                
        else:
            # General enhancement
            factor = 1.0 + base_strength * 0.4
        
        # Apply block-specific modulation (minimal)
        if 'mid' in block:
            factor *= 1.0 + base_strength * 0.1  # Slight content boost
        
        self.activations.append({
            'step': step,
            'block': block,
            'sigma': float(sigma),
            'factor': float(factor),
            'content_type': 'portrait' if self.is_portrait else 'landscape' if self.is_landscape else 'object' if self.is_object else 'general'
        })
        
        return out * factor


class ConsistencyEnforcer:
    """
    Processor that enforces consistency by limiting deviation from baseline.
    Uses momentum to prevent dramatic oscillations.
    """
    
    def __init__(self, max_deviation: float = 0.2):
        self.max_deviation = max_deviation
        self.momentum_history = []
        self.activations = []
        
    def __call__(self, *, out=None, meta: Dict[str, Any]) -> Optional[Any]:
        if out is None:
            return None
            
        sigma = meta.get('sigma', 0.0)
        block = meta.get('block_id', '')
        step = meta.get('step_idx', 0)
        
        # Calculate enhancement with momentum dampening
        base_enhancement = 1.0 + self.max_deviation * 0.5  # 50% of max deviation
        
        # Apply momentum dampening (prevents oscillations)
        if len(self.momentum_history) > 0:
            avg_momentum = np.mean(self.momentum_history[-3:])  # Last 3 steps
            dampening = max(0.5, 1.0 - abs(avg_momentum - 1.0))  # Reduce if deviating
            factor = 1.0 + (base_enhancement - 1.0) * dampening
        else:
            factor = base_enhancement
        
        # Clamp to prevent extreme values
        factor = max(0.8, min(1.2, factor))
        
        # Update momentum history
        self.momentum_history.append(factor)
        if len(self.momentum_history) > 10:
            self.momentum_history.pop(0)
        
        self.activations.append({
            'step': step,
            'block': block,
            'sigma': float(sigma),
            'factor': float(factor),
            'momentum_avg': float(np.mean(self.momentum_history)) if self.momentum_history else 1.0
        })
        
        return out * factor


def create_stabilized_comparison(prompt: str, processors: Dict[str, Any], 
                               base_seed: int = 42) -> List[Tuple[Image.Image, str, Dict]]:
    """Create stabilized comparison with multiple enhancement approaches."""
    
    print(f"\n📊 Stabilized Comparison: {prompt}")
    
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    results = []
    
    def cleanup():
        if attn_hooks.ATTN_HOOKS_ENABLED:
            attn_hooks.attention_registry.clear()
            attn_hooks.disable_hooks()
        mx.clear_cache()
    
    def generate_single(prompt: str, processor=None, seed: int = 42):
        mx.random.seed(seed)
        
        if processor:
            attn_hooks.enable_hooks()
            for block in ['down_1', 'mid', 'up_1']:  # Reduced blocks for stability
                attn_hooks.register_processor(block, processor)
        
        latents = sd.generate_latents(prompt, n_images=1, cfg_weight=7.5, num_steps=12, seed=seed)
        
        x_t = None
        for x in latents:
            x_t = x
            mx.eval(x_t)
        
        image = sd.decode(x_t)
        mx.eval(image)
        
        img_array = (image[0] * 255).astype(mx.uint8)
        pil_image = Image.fromarray(np.array(img_array))
        
        cleanup()
        return pil_image
    
    # Generate baseline
    print("  Normal...")
    normal_img = generate_single(prompt, None, base_seed)
    results.append((normal_img, "Normal", {}))
    
    # Generate with each processor
    for proc_name, processor in processors.items():
        print(f"  {proc_name}...")
        enhanced_img = generate_single(prompt, processor, base_seed)  # Same seed for consistency
        
        stats = {
            'activations': len(processor.activations) if hasattr(processor, 'activations') else 0,
            'avg_factor': np.mean([a['factor'] for a in processor.activations]) if hasattr(processor, 'activations') and processor.activations else 1.0
        }
        
        results.append((enhanced_img, proc_name, stats))
    
    return results


def create_stability_grid(results: List[Tuple[Image.Image, str, Dict]], 
                         prompt: str, output_name: str):
    """Create grid showing stability improvements."""
    
    if not results:
        return
    
    img_width, img_height = results[0][0].size
    num_versions = len(results)
    
    padding = 15
    text_height = 40
    title_height = 60
    
    grid_width = num_versions * img_width + (num_versions + 1) * padding
    grid_height = img_height + text_height + title_height + padding * 2
    
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font_title = font_label = ImageFont.load_default()
    
    # Title
    title = f"Stabilized CorePulse: {prompt[:40]}..."
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((grid_width - title_width) // 2, padding), title, fill='black', font=font_title)
    
    # Images and labels
    y_img = title_height + padding
    y_label = y_img + img_height + 5
    
    for i, (image, label, stats) in enumerate(results):
        x = padding + i * (img_width + padding)
        
        # Image
        grid.paste(image, (x, y_img))
        
        # Label
        color = 'gray' if label == "Normal" else 'blue'
        draw.text((x + 5, y_label), label, fill=color, font=font_label)
        
        # Stats
        if stats.get('avg_factor'):
            factor_text = f"×{stats['avg_factor']:.2f}"
            draw.text((x + 5, y_label + 15), factor_text, fill='green', font=font_label)
    
    grid.save(output_name)
    print(f"✅ Saved: {output_name}")


def main():
    """Run stabilized CorePulse demonstration."""
    
    print("\n" + "🎯"*50)
    print("   STABILIZED COREPULSE - CONTROLLED ENHANCEMENT")
    print("🎯"*50)
    
    # Test prompts that caused oscillations before
    test_cases = [
        "a professional business portrait",
        "a magical fantasy landscape", 
        "a futuristic sports car"
    ]
    
    # Define stabilized processors
    processors = {
        'Stabilized': StabilizedProcessor('photorealistic', strength=0.12),
        'Semantic': SemanticPreservationProcessor("business portrait", "quality"),
        'Consistent': ConsistencyEnforcer(max_deviation=0.15)
    }
    
    for i, prompt in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Testing stabilization...")
        
        # Update semantic processor for each prompt
        processors['Semantic'] = SemanticPreservationProcessor(prompt, "quality")
        
        # Generate comparison
        results = create_stabilized_comparison(prompt, processors, base_seed=5000 + i * 100)
        
        # Create grid
        safe_prompt = re.sub(r'[^\w\s-]', '', prompt).replace(' ', '_')[:25]
        grid_name = f"stabilized_{i:02d}_{safe_prompt}.png"
        create_stability_grid(results, prompt, grid_name)
        
        # Analysis
        print(f"  Analysis:")
        for img, name, stats in results[1:]:  # Skip normal
            avg_factor = stats.get('avg_factor', 1.0)
            activations = stats.get('activations', 0)
            deviation = abs(avg_factor - 1.0)
            print(f"    {name}: ×{avg_factor:.3f} deviation ({deviation:.1%}), {activations} activations")
    
    print(f"\n{'='*70}")
    print("🎉 STABILIZED COREPULSE COMPLETE!")
    print(f"{'='*70}")
    print("Improvements:")
    print("  ✓ Gentle enhancement multipliers (×1.05-1.15 instead of ×1.3-1.6)")
    print("  ✓ Semantic preservation based on prompt content")
    print("  ✓ Momentum dampening to prevent oscillations")
    print("  ✓ Consistency enforcement with deviation limits")
    print("  ✓ Block-specific modulation for targeted enhancement")
    print("\n🎯 CorePulse now provides controlled, predictable enhancements!")


if __name__ == "__main__":
    main()