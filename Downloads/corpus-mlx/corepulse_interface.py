#!/usr/bin/env python3
"""
User-friendly interface for CorePulse MLX.
Provides easy-to-use API for common diffusion control tasks.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from PIL import Image
import mlx.core as mx
import json

# Add MLX examples to path
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples"))
sys.path.insert(0, str(Path(__file__).parent / "mlx-examples/stable_diffusion"))

from stable_diffusion import StableDiffusion
from stable_diffusion import attn_hooks
from stable_diffusion import sigma_hooks


class ControlMode(Enum):
    """Available control modes."""
    STRUCTURE = "structure"  # Early/down blocks
    CONTENT = "content"      # Middle blocks
    DETAIL = "detail"        # Late/up blocks
    BALANCED = "balanced"    # All blocks equally
    CUSTOM = "custom"        # User-defined


class EffectStrength(Enum):
    """Preset effect strengths."""
    SUBTLE = 0.1
    LIGHT = 0.3
    MEDIUM = 0.5
    STRONG = 0.8
    EXTREME = 1.2


@dataclass
class ControlPoint:
    """Single control point in generation."""
    sigma_min: float
    sigma_max: float
    blocks: List[str]
    weight: float
    description: str = ""


@dataclass
class ControlSchedule:
    """Full control schedule for generation."""
    points: List[ControlPoint] = field(default_factory=list)
    
    def add_point(self, sigma_range: Tuple[float, float], 
                  blocks: Union[str, List[str]], 
                  weight: float,
                  description: str = ""):
        """Add control point to schedule."""
        if isinstance(blocks, str):
            blocks = [blocks]
        point = ControlPoint(sigma_range[0], sigma_range[1], blocks, weight, description)
        self.points.append(point)
        return self
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            'points': [
                {
                    'sigma_range': [p.sigma_min, p.sigma_max],
                    'blocks': p.blocks,
                    'weight': p.weight,
                    'description': p.description
                }
                for p in self.points
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ControlSchedule':
        """Create from dictionary."""
        schedule = cls()
        for p in data['points']:
            schedule.add_point(
                tuple(p['sigma_range']),
                p['blocks'],
                p['weight'],
                p.get('description', '')
            )
        return schedule


class ScheduleProcessor:
    """Processor that follows a control schedule."""
    
    def __init__(self, schedule: ControlSchedule):
        self.schedule = schedule
        self.activations = []
        
    def __call__(self, *, out=None, meta: Dict[str, Any]) -> Optional[Any]:
        if out is None:
            return None
            
        block = meta.get('block_id', '')
        sigma = meta.get('sigma', 0.0)
        step = meta.get('step_idx', 0)
        
        # Find applicable control points
        for point in self.schedule.points:
            if point.sigma_min <= sigma <= point.sigma_max:
                if any(b in block for b in point.blocks):
                    self.activations.append({
                        'step': step,
                        'block': block,
                        'sigma': sigma,
                        'weight': point.weight
                    })
                    return out * point.weight
                    
        return out


class CorePulseInterface:
    """
    User-friendly interface for CorePulse.
    High-level API for common use cases.
    """
    
    def __init__(self, model_path: str = "stabilityai/stable-diffusion-2-1-base"):
        self.sd = StableDiffusion(model_path, float16=True)
        self.current_schedule = None
        self.presets = self._load_presets()
        
    def _load_presets(self) -> Dict[str, ControlSchedule]:
        """Load preset control schedules."""
        presets = {}
        
        # Photorealistic preset
        photorealistic = ControlSchedule()
        photorealistic.add_point((10, 15), ['down'], 1.2, "Strong structure")
        photorealistic.add_point((5, 10), ['mid'], 1.4, "Enhanced content")
        photorealistic.add_point((0, 5), ['up'], 1.3, "Detailed refinement")
        presets['photorealistic'] = photorealistic
        
        # Artistic preset
        artistic = ControlSchedule()
        artistic.add_point((10, 15), ['down'], 0.8, "Loose structure")
        artistic.add_point((5, 10), ['mid'], 1.3, "Creative content")
        artistic.add_point((0, 5), ['up'], 1.5, "Artistic details")
        presets['artistic'] = artistic
        
        # Abstract preset
        abstract = ControlSchedule()
        abstract.add_point((10, 15), ['down'], 0.6, "Minimal structure")
        abstract.add_point((5, 10), ['mid'], 1.6, "Abstract patterns")
        abstract.add_point((0, 5), ['up'], 1.2, "Pattern details")
        presets['abstract'] = abstract
        
        # Sharp preset
        sharp = ControlSchedule()
        sharp.add_point((10, 15), ['down'], 1.4, "Crisp structure")
        sharp.add_point((5, 10), ['mid'], 1.2, "Clear content")
        sharp.add_point((0, 5), ['up'], 1.5, "Sharp details")
        presets['sharp'] = sharp
        
        # Soft preset
        soft = ControlSchedule()
        soft.add_point((10, 15), ['down'], 0.9, "Soft structure")
        soft.add_point((5, 10), ['mid'], 1.0, "Smooth content")
        soft.add_point((0, 5), ['up'], 0.8, "Gentle details")
        presets['soft'] = soft
        
        return presets
    
    def generate_simple(self, 
                       prompt: str,
                       mode: ControlMode = ControlMode.BALANCED,
                       strength: EffectStrength = EffectStrength.MEDIUM,
                       **kwargs) -> Image.Image:
        """
        Simple generation with basic control mode.
        
        Args:
            prompt: Text prompt
            mode: Control mode (structure/content/detail/balanced)
            strength: Effect strength
            **kwargs: Additional generation parameters
        """
        
        # Build schedule based on mode
        schedule = ControlSchedule()
        weight = strength.value
        
        if mode == ControlMode.STRUCTURE:
            schedule.add_point((10, 15), ['down_0', 'down_1', 'down_2'], 1 + weight)
            schedule.add_point((0, 10), ['mid', 'up_0', 'up_1', 'up_2'], 1.0)
            
        elif mode == ControlMode.CONTENT:
            schedule.add_point((5, 12), ['mid'], 1 + weight)
            schedule.add_point((0, 5), ['down', 'up'], 1.0)
            schedule.add_point((12, 15), ['down', 'up'], 1.0)
            
        elif mode == ControlMode.DETAIL:
            schedule.add_point((0, 5), ['up_0', 'up_1', 'up_2'], 1 + weight)
            schedule.add_point((5, 15), ['down', 'mid'], 1.0)
            
        else:  # BALANCED
            schedule.add_point((0, 15), ['down', 'mid', 'up'], 1 + weight * 0.5)
        
        return self.generate_with_schedule(prompt, schedule, **kwargs)
    
    def generate_preset(self, 
                       prompt: str,
                       preset: str,
                       strength: float = 1.0,
                       **kwargs) -> Image.Image:
        """
        Generate using a preset control schedule.
        
        Args:
            prompt: Text prompt
            preset: Preset name ('photorealistic', 'artistic', 'abstract', 'sharp', 'soft')
            strength: Strength multiplier for preset
            **kwargs: Additional generation parameters
        """
        
        if preset not in self.presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.presets.keys())}")
        
        # Get preset and apply strength
        schedule = self.presets[preset]
        if strength != 1.0:
            for point in schedule.points:
                point.weight = 1 + (point.weight - 1) * strength
        
        return self.generate_with_schedule(prompt, schedule, **kwargs)
    
    def generate_with_schedule(self,
                              prompt: str,
                              schedule: ControlSchedule,
                              **kwargs) -> Image.Image:
        """
        Generate with custom control schedule.
        
        Args:
            prompt: Text prompt
            schedule: ControlSchedule object
            **kwargs: Additional generation parameters
        """
        
        # Create processor from schedule
        processor = ScheduleProcessor(schedule)
        
        # Enable hooks
        attn_hooks.enable_hooks()
        
        # Register processor for all relevant blocks
        blocks_set = set()
        for point in schedule.points:
            blocks_set.update(point.blocks)
        
        # Expand block patterns to actual block names
        actual_blocks = []
        for pattern in blocks_set:
            if pattern == 'down':
                actual_blocks.extend(['down_0', 'down_1', 'down_2'])
            elif pattern == 'mid':
                actual_blocks.append('mid')
            elif pattern == 'up':
                actual_blocks.extend(['up_0', 'up_1', 'up_2'])
            elif pattern in ['down_0', 'down_1', 'down_2', 'mid', 'up_0', 'up_1', 'up_2']:
                actual_blocks.append(pattern)
        
        for block in set(actual_blocks):
            attn_hooks.register_processor(block, processor)
        
        # Generate
        image = self._generate(prompt, **kwargs)
        
        # Store schedule and activations
        self.current_schedule = schedule
        self.last_activations = processor.activations
        
        # Clean up
        attn_hooks.attention_registry.clear()
        attn_hooks.disable_hooks()
        
        return image
    
    def create_variations(self,
                         prompt: str,
                         num_variations: int = 4,
                         variation_strength: float = 0.3,
                         **kwargs) -> List[Image.Image]:
        """
        Create variations of a prompt with different control settings.
        
        Args:
            prompt: Text prompt
            num_variations: Number of variations to create
            variation_strength: How different variations should be (0-1)
            **kwargs: Additional generation parameters
        """
        
        images = []
        base_seed = kwargs.get('seed', 42)
        
        for i in range(num_variations):
            # Vary the control mode
            if i == 0:
                mode = ControlMode.BALANCED
            elif i == 1:
                mode = ControlMode.STRUCTURE
            elif i == 2:
                mode = ControlMode.CONTENT
            else:
                mode = ControlMode.DETAIL
            
            # Vary the strength
            strength_val = 0.3 + variation_strength * (i / max(1, num_variations - 1))
            strength = EffectStrength(min(1.2, strength_val))
            
            # Use different seed for each
            kwargs['seed'] = base_seed + i * 100
            
            image = self.generate_simple(prompt, mode, strength, **kwargs)
            images.append(image)
            
        return images
    
    def interpolate(self,
                   prompt_start: str,
                   prompt_end: str,
                   steps: int = 5,
                   **kwargs) -> List[Image.Image]:
        """
        Interpolate between two prompts.
        
        Args:
            prompt_start: Starting prompt
            prompt_end: Ending prompt
            steps: Number of interpolation steps
            **kwargs: Additional generation parameters
        """
        
        images = []
        
        for i in range(steps):
            t = i / max(1, steps - 1)
            
            # Create interpolation schedule
            schedule = ControlSchedule()
            
            # Early steps favor start
            schedule.add_point((12, 15), ['down'], 1 + (1 - t) * 0.5, f"Start structure ({1-t:.1f})")
            
            # Middle blends
            schedule.add_point((6, 12), ['mid'], 1 + 0.3, "Blended content")
            
            # Late steps favor end
            schedule.add_point((0, 6), ['up'], 1 + t * 0.5, f"End details ({t:.1f})")
            
            # Use appropriate prompt
            if t < 0.5:
                prompt = prompt_start
            else:
                prompt = prompt_end
            
            kwargs['seed'] = kwargs.get('seed', 42) + i * 50
            image = self.generate_with_schedule(prompt, schedule, **kwargs)
            images.append(image)
            
        return images
    
    def save_schedule(self, schedule: ControlSchedule, filepath: str):
        """Save control schedule to JSON."""
        with open(filepath, 'w') as f:
            json.dump(schedule.to_dict(), f, indent=2)
    
    def load_schedule(self, filepath: str) -> ControlSchedule:
        """Load control schedule from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return ControlSchedule.from_dict(data)
    
    def get_activation_summary(self) -> Dict:
        """Get summary of last generation's activations."""
        if not hasattr(self, 'last_activations'):
            return {}
            
        summary = {
            'total_activations': len(self.last_activations),
            'blocks_activated': {},
            'sigma_range': [float('inf'), float('-inf')],
            'steps_range': [float('inf'), float('-inf')]
        }
        
        for act in self.last_activations:
            block = act['block']
            if block not in summary['blocks_activated']:
                summary['blocks_activated'][block] = 0
            summary['blocks_activated'][block] += 1
            
            summary['sigma_range'][0] = min(summary['sigma_range'][0], act['sigma'])
            summary['sigma_range'][1] = max(summary['sigma_range'][1], act['sigma'])
            
            summary['steps_range'][0] = min(summary['steps_range'][0], act['step'])
            summary['steps_range'][1] = max(summary['steps_range'][1], act['step'])
        
        return summary
    
    def _generate(self, prompt: str, **kwargs) -> Image.Image:
        """Internal generation helper."""
        seed = kwargs.get('seed', 42)
        mx.random.seed(seed)
        
        latents = self.sd.generate_latents(
            prompt,
            n_images=1,
            cfg_weight=kwargs.get('cfg_weight', 7.5),
            num_steps=kwargs.get('num_steps', 20),
            seed=seed
        )
        
        x_t = None
        for x in latents:
            x_t = x
            mx.eval(x_t)
        
        image = self.sd.decode(x_t)
        mx.eval(image)
        
        img_array = (image[0] * 255).astype(mx.uint8)
        return Image.fromarray(np.array(img_array))


def demo_simple_api():
    """Demonstrate the simple API."""
    print("\n" + "="*70)
    print("🎯 SIMPLE API DEMO")
    print("="*70)
    
    cp = CorePulseInterface()
    prompt = "a magical forest with glowing mushrooms"
    
    # Generate with different modes
    modes = [
        (ControlMode.BALANCED, "Balanced"),
        (ControlMode.STRUCTURE, "Structure Focus"),
        (ControlMode.CONTENT, "Content Focus"),
        (ControlMode.DETAIL, "Detail Focus")
    ]
    
    print(f"\nPrompt: {prompt}")
    
    for mode, name in modes:
        print(f"\n{name}:")
        img = cp.generate_simple(
            prompt, 
            mode=mode, 
            strength=EffectStrength.MEDIUM,
            num_steps=15,
            seed=100
        )
        img.save(f"interface_{mode.value}.png")
        print(f"  ✅ Saved: interface_{mode.value}.png")


def demo_presets():
    """Demonstrate preset styles."""
    print("\n" + "="*70)
    print("🎨 PRESETS DEMO")
    print("="*70)
    
    cp = CorePulseInterface()
    prompt = "a portrait of a wise wizard"
    
    presets = ['photorealistic', 'artistic', 'abstract', 'sharp', 'soft']
    
    print(f"\nPrompt: {prompt}")
    
    for preset in presets:
        print(f"\n{preset.title()}:")
        img = cp.generate_preset(
            prompt,
            preset=preset,
            strength=1.0,
            num_steps=15,
            seed=200
        )
        img.save(f"interface_preset_{preset}.png")
        print(f"  ✅ Saved: interface_preset_{preset}.png")


def demo_variations():
    """Demonstrate variation generation."""
    print("\n" + "="*70)
    print("🔀 VARIATIONS DEMO")
    print("="*70)
    
    cp = CorePulseInterface()
    prompt = "a steampunk airship"
    
    print(f"\nCreating variations of: {prompt}")
    
    images = cp.create_variations(
        prompt,
        num_variations=4,
        variation_strength=0.5,
        num_steps=15,
        seed=300
    )
    
    for i, img in enumerate(images):
        img.save(f"interface_variation_{i}.png")
        print(f"  ✅ Saved: interface_variation_{i}.png")


def demo_interpolation():
    """Demonstrate prompt interpolation."""
    print("\n" + "="*70)
    print("🌈 INTERPOLATION DEMO")
    print("="*70)
    
    cp = CorePulseInterface()
    
    prompt_start = "a peaceful zen garden"
    prompt_end = "a bustling cyberpunk city"
    
    print(f"\nInterpolating:")
    print(f"  From: {prompt_start}")
    print(f"  To: {prompt_end}")
    
    images = cp.interpolate(
        prompt_start,
        prompt_end,
        steps=5,
        num_steps=15,
        seed=400
    )
    
    for i, img in enumerate(images):
        img.save(f"interface_interpolate_{i}.png")
        print(f"  ✅ Step {i+1}/5 saved")


def demo_custom_schedule():
    """Demonstrate custom control schedules."""
    print("\n" + "="*70)
    print("📋 CUSTOM SCHEDULE DEMO")
    print("="*70)
    
    cp = CorePulseInterface()
    prompt = "an ancient temple ruins"
    
    # Create custom schedule
    print("\nCreating custom control schedule...")
    schedule = ControlSchedule()
    
    # Phase 1: Strong architectural structure
    schedule.add_point((12, 15), ['down_0', 'down_1'], 1.5, "Architecture phase")
    
    # Phase 2: Historical atmosphere
    schedule.add_point((7, 12), ['mid'], 1.3, "Atmosphere phase")
    
    # Phase 3: Weathering and details
    schedule.add_point((0, 7), ['up_1', 'up_2'], 1.4, "Weathering details")
    
    print("Schedule phases:")
    for i, point in enumerate(schedule.points, 1):
        print(f"  {i}. σ[{point.sigma_min:.0f}-{point.sigma_max:.0f}]: {point.description}")
    
    # Generate with custom schedule
    print("\nGenerating with custom schedule...")
    img = cp.generate_with_schedule(prompt, schedule, num_steps=20, seed=500)
    img.save("interface_custom_schedule.png")
    print("  ✅ Saved: interface_custom_schedule.png")
    
    # Save schedule for reuse
    cp.save_schedule(schedule, "custom_temple_schedule.json")
    print("  ✅ Schedule saved: custom_temple_schedule.json")
    
    # Get activation summary
    summary = cp.get_activation_summary()
    print("\nActivation summary:")
    print(f"  Total activations: {summary.get('total_activations', 0)}")
    print(f"  Sigma range: {summary.get('sigma_range', [])}")
    print(f"  Blocks activated: {summary.get('blocks_activated', {})}")


def main():
    """Run all interface demos."""
    print("\n" + "🚀"*35)
    print("   COREPULSE INTERFACE DEMOS")
    print("🚀"*35)
    
    try:
        demo_simple_api()
        demo_presets()
        demo_variations()
        demo_interpolation()
        demo_custom_schedule()
        
        print("\n" + "="*70)
        print("✅ ALL INTERFACE DEMOS COMPLETE!")
        print("="*70)
        print("\nInterface features demonstrated:")
        print("  ✓ Simple API with control modes")
        print("  ✓ Preset style templates")
        print("  ✓ Automatic variations")
        print("  ✓ Prompt interpolation")
        print("  ✓ Custom control schedules")
        print("\n🎉 CorePulse MLX interface is ready for use!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()