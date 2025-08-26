#!/usr/bin/env python3
"""
CorePulse-Enhanced Product Placement
Combines zero-hallucination product placement with CorePulse's advanced control
"""

import numpy as np
from PIL import Image, ImageFilter
import mlx.core as mx
from stable_diffusion import StableDiffusionXL
from corepulse_mlx import (
    CorePulseMLX,
    PromptInjection,
    InjectionLevel,
    TokenMask,
    SpatialInjection
)
from typing import Tuple, Optional, Union
from pathlib import Path
import cv2


class CorePulseProductPlacement:
    """
    Advanced product placement using CorePulse control
    - Zero hallucination on products
    - Multi-level scene control
    - Spatial region targeting
    - Token emphasis for perfect integration
    """
    
    def __init__(self, model_type: str = "sdxl", float16: bool = True):
        # Initialize base model
        if model_type == "sdxl":
            self.base_model = StableDiffusionXL(
                model="stabilityai/sdxl-turbo",
                float16=float16
            )
        else:
            raise ValueError("Only SDXL supported for now")
        
        print("Loading model...")
        self.base_model.ensure_models_are_loaded()
        
        # Wrap with CorePulse
        self.corepulse = CorePulseMLX(self.base_model)
        print("CorePulse initialized!")
    
    def extract_product(
        self,
        image_path: Union[str, Path],
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        tolerance: int = 30
    ) -> Tuple[Image.Image, Image.Image]:
        """Extract product with alpha mask"""
        img = Image.open(image_path).convert("RGBA")
        img_array = np.array(img)
        
        # Create mask from background
        rgb = img_array[:, :, :3]
        bg = np.array(bg_color)
        diff = np.abs(rgb - bg)
        distance = np.sqrt(np.sum(diff ** 2, axis=2))
        
        alpha = np.where(distance > tolerance, 255, 0).astype(np.uint8)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        
        product_rgb = Image.fromarray(img_array[:, :, :3], 'RGB')
        alpha_mask = Image.fromarray(alpha, 'L')
        
        return product_rgb, alpha_mask
    
    def create_enhanced_shadow(
        self,
        mask: Image.Image,
        canvas_size: Tuple[int, int],
        position: Tuple[int, int],
        light_direction: str = "top_left"
    ) -> Image.Image:
        """Create directional shadow based on lighting"""
        shadow_composite = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        
        # Determine shadow offset based on light direction
        shadow_offsets = {
            "top_left": [(10, 10), (15, 15), (25, 25)],
            "top_right": [(-10, 10), (-15, 15), (-25, 25)],
            "bottom_left": [(10, -10), (15, -15), (25, -25)],
            "bottom_right": [(-10, -10), (-15, -15), (-25, -25)]
        }
        
        offsets = shadow_offsets.get(light_direction, shadow_offsets["top_left"])
        blurs = [3, 8, 20]
        opacities = [180, 120, 60]
        
        for offset, blur, opacity in zip(offsets, blurs, opacities):
            shadow_layer = Image.new('L', canvas_size, 0)
            shadow_pos = (position[0] + offset[0], position[1] + offset[1])
            shadow_layer.paste(mask, shadow_pos)
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur))
            
            layer = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
            layer.paste((0, 0, 0, opacity), mask=shadow_layer)
            shadow_composite = Image.alpha_composite(shadow_composite, layer)
        
        return shadow_composite
    
    def place_with_corepulse(
        self,
        product_path: Union[str, Path],
        scene_description: str,
        output_path: Union[str, Path],
        # CorePulse controls
        surface_type: str = "table",  # table, floor, wall, shelf
        lighting_style: str = "natural",  # natural, studio, dramatic, soft
        environment_mood: str = "modern",  # modern, vintage, minimal, luxury
        # Placement controls
        position: Optional[Tuple[int, int]] = None,
        scale: float = 1.0,
        add_reflection: bool = False,
        # Generation params
        num_steps: int = 4,
        seed: Optional[int] = None,
        output_size: Tuple[int, int] = (1024, 1024)
    ) -> Image.Image:
        """
        Generate scene with CorePulse control and place product
        
        Uses multi-level injection for:
        - Surface in early blocks (structure)
        - Environment in mid blocks (context)
        - Lighting in late blocks (style)
        """
        
        print(f"\n{'='*60}")
        print("COREPULSE PRODUCT PLACEMENT")
        print(f"{'='*60}")
        
        # Extract product
        print(f"1. Extracting product from {product_path}")
        product_rgb, mask = self.extract_product(product_path)
        
        # Scale product
        if scale != 1.0:
            new_size = (int(product_rgb.width * scale), int(product_rgb.height * scale))
            product_rgb = product_rgb.resize(new_size, Image.Resampling.LANCZOS)
            mask = mask.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create multi-level prompts
        print("\n2. Creating CorePulse injections:")
        
        # Base prompt combines all elements
        base_prompt = f"{scene_description}, {surface_type} surface, {lighting_style} lighting, {environment_mood} environment"
        
        # Multi-level injections
        injections = [
            # Structure/Surface (early blocks)
            PromptInjection(
                prompt=f"clean {surface_type} surface, proper perspective, grounded placement",
                levels=[InjectionLevel.ENCODER_EARLY],
                strength=0.9
            ),
            # Environment/Context (mid blocks)
            PromptInjection(
                prompt=f"{environment_mood} {scene_description}",
                levels=[InjectionLevel.ENCODER_MID, InjectionLevel.DECODER_MID],
                strength=0.8
            ),
            # Lighting/Style (late blocks)
            PromptInjection(
                prompt=f"{lighting_style} lighting, realistic shadows, ambient occlusion",
                levels=[InjectionLevel.ENCODER_LATE, InjectionLevel.DECODER_LATE],
                strength=0.7
            )
        ]
        
        print(f"   - Surface control: {surface_type} (early blocks)")
        print(f"   - Environment: {environment_mood} (mid blocks)")
        print(f"   - Lighting: {lighting_style} (late blocks)")
        
        # Token emphasis for key elements
        token_masks = [
            TokenMask(
                tokens=[surface_type],
                mask_type="amplify",
                strength=1.5
            ),
            TokenMask(
                tokens=["shadow", "reflection"] if add_reflection else ["shadow"],
                mask_type="amplify",
                strength=1.2
            )
        ]
        
        # Determine position if not specified
        if position is None:
            # Smart positioning based on surface type
            if surface_type == "floor":
                position = (
                    (output_size[0] - product_rgb.width) // 2,
                    int(output_size[1] * 0.7)
                )
            elif surface_type == "wall":
                position = (
                    (output_size[0] - product_rgb.width) // 2,
                    int(output_size[1] * 0.3)
                )
            else:  # table, shelf
                position = (
                    (output_size[0] - product_rgb.width) // 2,
                    int(output_size[1] * 0.6)
                )
        
        # Spatial injection for product area
        product_bbox = (
            position[0],
            position[1],
            position[0] + product_rgb.width,
            position[1] + product_rgb.height
        )
        
        spatial_injections = [
            SpatialInjection(
                prompt=f"clear space for product, {surface_type} texture",
                bbox=product_bbox,
                strength=0.6,
                feather=20
            )
        ]
        
        # Generate scene with CorePulse control
        print("\n3. Generating scene with CorePulse control...")
        print(f"   Seed: {seed if seed else 'random'}")
        
        result = self.corepulse.generate_with_control(
            base_prompt=base_prompt,
            prompt_injections=injections,
            token_masks=token_masks,
            spatial_injections=spatial_injections,
            negative_prompt="blurry, distorted, unrealistic, floating",
            num_steps=num_steps,
            cfg_weight=0.0,  # SDXL Turbo
            seed=seed,
            output_size=output_size
        )
        
        # Convert to PIL
        print("\n4. Processing generated scene...")
        img_array = np.array(result)
        if img_array.ndim == 4:
            img_array = img_array[0]
        if img_array.shape[0] == 3:
            img_array = np.transpose(img_array, (1, 2, 0))
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        scene = Image.fromarray(img_array).resize(output_size, Image.Resampling.LANCZOS)
        
        # Add shadow
        print("\n5. Adding realistic shadow...")
        light_dir = "top_left" if lighting_style == "natural" else "top_right"
        shadow = self.create_enhanced_shadow(mask, output_size, position, light_dir)
        scene = Image.alpha_composite(scene.convert('RGBA'), shadow).convert('RGB')
        
        # Add reflection if requested
        if add_reflection and surface_type in ["table", "floor"]:
            print("\n6. Adding reflection...")
            reflection = product_rgb.transpose(Image.FLIP_TOP_BOTTOM)
            refl_mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Create gradient fade
            gradient = Image.new('L', refl_mask.size, 0)
            for y in range(gradient.height):
                opacity = int(255 * (1 - y / gradient.height) * 0.3)
                gradient.paste(opacity, (0, y, gradient.width, y+1))
            
            refl_mask = Image.composite(refl_mask, Image.new('L', refl_mask.size, 0), gradient)
            
            refl_pos = (position[0], position[1] + product_rgb.height + 2)
            scene.paste(reflection, refl_pos, refl_mask)
        
        # Place product (zero hallucination - preserving original pixels)
        print("\n7. Placing product with zero hallucination...")
        scene.paste(product_rgb, position, mask)
        
        # Save result
        scene.save(output_path)
        print(f"\n✅ Saved to: {output_path}")
        
        print(f"\n{'='*60}")
        print("COREPULSE PRODUCT PLACEMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Product placed with:")
        print(f"  • Zero hallucination (original preserved)")
        print(f"  • Multi-level scene control")
        print(f"  • Proper {surface_type} grounding")
        print(f"  • {lighting_style} lighting with shadows")
        print(f"  • {environment_mood} environment")
        
        return scene


def main():
    """Test CorePulse product placement"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CorePulse-enhanced product placement")
    parser.add_argument("--product", default="test_product_watch.png", help="Product image")
    parser.add_argument("--scene", default="modern office desk with laptop", help="Scene description")
    parser.add_argument("--surface", default="table", choices=["table", "floor", "wall", "shelf"])
    parser.add_argument("--lighting", default="natural", choices=["natural", "studio", "dramatic", "soft"])
    parser.add_argument("--mood", default="modern", choices=["modern", "vintage", "minimal", "luxury"])
    parser.add_argument("--output", default="corepulse_placement.png", help="Output path")
    parser.add_argument("--scale", type=float, default=0.8, help="Product scale")
    parser.add_argument("--reflection", action="store_true", help="Add reflection")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CorePulseProductPlacement(model_type="sdxl", float16=True)
    
    # Generate with CorePulse control
    pipeline.place_with_corepulse(
        product_path=args.product,
        scene_description=args.scene,
        output_path=args.output,
        surface_type=args.surface,
        lighting_style=args.lighting,
        environment_mood=args.mood,
        scale=args.scale,
        add_reflection=args.reflection,
        seed=args.seed
    )


if __name__ == "__main__":
    main()