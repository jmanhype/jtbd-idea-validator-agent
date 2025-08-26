#!/usr/bin/env python3
"""
Improved Product Placement Pipeline V2
Better grounding, shadows, and integration
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import cv2
import mlx.core as mx
from stable_diffusion import StableDiffusion, StableDiffusionXL
from typing import Optional, Tuple, Union
from pathlib import Path


class ImprovedProductPlacement:
    """
    Version 2 with improved integration:
    - Better surface detection
    - Perspective-aware placement
    - Multi-layer shadows
    - Edge feathering
    - Reflection generation (for glossy surfaces)
    """
    
    def __init__(self, model_type: str = "sdxl", float16: bool = True):
        if model_type == "sdxl":
            self.model = StableDiffusionXL(
                model="stabilityai/sdxl-turbo",
                float16=float16
            )
            self.is_sdxl = True
            self.default_steps = 4
            self.default_cfg = 0.0
        else:
            self.model = StableDiffusion(
                model="stabilityai/stable-diffusion-2-1-base",
                float16=float16
            )
            self.is_sdxl = False
            self.default_steps = 30
            self.default_cfg = 7.5
        
        print(f"Loading {model_type} model...")
        self.model.ensure_models_are_loaded()
        print("Model loaded!")
    
    def extract_product_advanced(
        self,
        image_path: Union[str, Path],
        method: str = "threshold",
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        tolerance: int = 30
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Advanced product extraction with better edge handling.
        
        Returns:
            Tuple of (product_rgb, alpha_mask, edge_mask)
        """
        
        img = Image.open(image_path).convert("RGBA")
        img_array = np.array(img)
        
        # Check if image has alpha channel
        if img_array.shape[2] == 4 and not np.all(img_array[:, :, 3] == 255):
            # Use existing alpha
            alpha = img_array[:, :, 3]
        else:
            # Create alpha from background color
            rgb = img_array[:, :, :3]
            
            # Calculate distance from background color
            bg = np.array(bg_color)
            diff = np.abs(rgb - bg)
            distance = np.sqrt(np.sum(diff ** 2, axis=2))
            
            # Create mask
            alpha = np.where(distance > tolerance, 255, 0).astype(np.uint8)
            
            # Clean up with morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Smooth edges
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        
        # Create edge mask for better blending
        edges = cv2.Canny(alpha, 50, 150)
        edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edge_mask = cv2.GaussianBlur(edge_mask, (5, 5), 0)
        
        # Convert to PIL
        product_rgb = Image.fromarray(img_array[:, :, :3], 'RGB')
        alpha_mask = Image.fromarray(alpha, 'L')
        edge_mask = Image.fromarray(edge_mask, 'L')
        
        return product_rgb, alpha_mask, edge_mask
    
    def create_layered_shadow(
        self,
        mask: Image.Image,
        canvas_size: Tuple[int, int],
        position: Tuple[int, int],
        shadow_params: dict
    ) -> Image.Image:
        """
        Create multi-layer shadow for more realism.
        
        Args:
            mask: Product mask
            canvas_size: Output canvas size
            position: Product position
            shadow_params: Shadow configuration
        
        Returns:
            Composite shadow layer
        """
        
        shadow_composite = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        
        # Contact shadow (dark, sharp, close)
        contact_shadow = Image.new('L', canvas_size, 0)
        contact_offset = (2, 2)
        contact_pos = (position[0] + contact_offset[0], position[1] + contact_offset[1])
        contact_shadow.paste(mask, contact_pos)
        contact_shadow = contact_shadow.filter(ImageFilter.GaussianBlur(radius=3))
        
        contact_layer = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        contact_layer.paste((0, 0, 0, 180), mask=contact_shadow)
        
        # Mid shadow (medium blur, medium distance)
        mid_shadow = Image.new('L', canvas_size, 0)
        mid_offset = shadow_params.get('offset', (8, 8))
        mid_pos = (position[0] + mid_offset[0], position[1] + mid_offset[1])
        mid_shadow.paste(mask, mid_pos)
        mid_shadow = mid_shadow.filter(ImageFilter.GaussianBlur(radius=8))
        
        mid_layer = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        mid_layer.paste((0, 0, 0, 120), mask=mid_shadow)
        
        # Soft shadow (very blurred, far)
        soft_shadow = Image.new('L', canvas_size, 0)
        soft_offset = (mid_offset[0] * 1.5, mid_offset[1] * 1.5)
        soft_pos = (position[0] + int(soft_offset[0]), position[1] + int(soft_offset[1]))
        soft_shadow.paste(mask, soft_pos)
        soft_shadow = soft_shadow.filter(ImageFilter.GaussianBlur(radius=20))
        
        soft_layer = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        soft_layer.paste((0, 0, 0, 60), mask=soft_shadow)
        
        # Composite all shadow layers
        shadow_composite = Image.alpha_composite(shadow_composite, soft_layer)
        shadow_composite = Image.alpha_composite(shadow_composite, mid_layer)
        shadow_composite = Image.alpha_composite(shadow_composite, contact_layer)
        
        # Adjust overall opacity
        opacity = shadow_params.get('opacity', 0.5)
        shadow_composite.putalpha(
            Image.eval(shadow_composite.split()[3], lambda x: int(x * opacity))
        )
        
        return shadow_composite
    
    def add_reflection(
        self,
        product: Image.Image,
        mask: Image.Image,
        strength: float = 0.3
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Create a reflection of the product for glossy surfaces.
        
        Args:
            product: Product image
            mask: Product mask
            strength: Reflection opacity (0-1)
        
        Returns:
            Tuple of (reflection_image, reflection_mask)
        """
        
        # Flip vertically
        reflection = product.transpose(Image.FLIP_TOP_BOTTOM)
        refl_mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Create gradient fade
        gradient = Image.new('L', refl_mask.size, 0)
        draw = ImageDraw.Draw(gradient)
        
        for y in range(gradient.height):
            opacity = int(255 * (1 - y / gradient.height) * strength)
            draw.rectangle([(0, y), (gradient.width, y+1)], fill=opacity)
        
        # Apply gradient to mask
        refl_mask = Image.composite(refl_mask, Image.new('L', refl_mask.size, 0), gradient)
        
        # Reduce reflection opacity
        enhancer = ImageEnhance.Brightness(reflection)
        reflection = enhancer.enhance(0.7)
        
        return reflection, refl_mask
    
    def adjust_product_lighting(
        self,
        product: Image.Image,
        scene: Image.Image,
        mask: Image.Image,
        position: Tuple[int, int]
    ) -> Image.Image:
        """
        Adjust product lighting to match scene ambiance.
        
        Args:
            product: Product image
            scene: Background scene
            mask: Product mask
            position: Product position in scene
        
        Returns:
            Lighting-adjusted product
        """
        
        # Sample colors around product position
        scene_array = np.array(scene)
        mask_array = np.array(mask) > 128
        
        # Get region around product
        y1 = max(0, position[1] - 50)
        y2 = min(scene.height, position[1] + product.height + 50)
        x1 = max(0, position[0] - 50)
        x2 = min(scene.width, position[0] + product.width + 50)
        
        region = scene_array[y1:y2, x1:x2]
        
        # Calculate average color temperature
        avg_color = np.mean(region, axis=(0, 1))
        
        # Calculate adjustment factors
        r_factor = avg_color[0] / 128
        g_factor = avg_color[1] / 128
        b_factor = avg_color[2] / 128
        
        # Limit adjustments
        r_factor = np.clip(r_factor, 0.7, 1.3)
        g_factor = np.clip(g_factor, 0.7, 1.3)
        b_factor = np.clip(b_factor, 0.7, 1.3)
        
        # Apply color adjustment
        product_array = np.array(product).astype(float)
        product_array[:, :, 0] *= r_factor
        product_array[:, :, 1] *= g_factor
        product_array[:, :, 2] *= b_factor
        product_array = np.clip(product_array, 0, 255).astype(np.uint8)
        
        # Adjust brightness
        brightness = np.mean(avg_color) / 128
        brightness = np.clip(brightness, 0.8, 1.2)
        
        adjusted = Image.fromarray(product_array)
        enhancer = ImageEnhance.Brightness(adjusted)
        adjusted = enhancer.enhance(brightness)
        
        return adjusted
    
    def composite_with_feathering(
        self,
        product: Image.Image,
        background: Image.Image,
        mask: Image.Image,
        edge_mask: Image.Image,
        position: Tuple[int, int]
    ) -> Image.Image:
        """
        Composite with edge feathering for smooth blending.
        
        Args:
            product: Product image
            background: Background scene
            mask: Product mask
            edge_mask: Edge mask for feathering
            position: Product position
        
        Returns:
            Composited image
        """
        
        result = background.copy()
        
        # Create feathered mask
        feathered_mask = mask.copy()
        feathered_mask = feathered_mask.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Paste with feathered edges
        result.paste(product, position, feathered_mask)
        
        return result
    
    def place_product_v2(
        self,
        product_path: Union[str, Path],
        scene_prompt: str,
        output_path: Union[str, Path],
        position: Optional[Tuple[int, int]] = None,
        scale: float = 1.0,
        add_reflection: bool = False,
        adjust_lighting: bool = True,
        num_steps: Optional[int] = None,
        cfg_weight: Optional[float] = None,
        seed: Optional[int] = None,
        output_size: Tuple[int, int] = (1024, 1024)
    ) -> Image.Image:
        """
        Improved product placement with better integration.
        """
        
        # Extract product with advanced method
        print("Extracting product...")
        product_rgb, mask, edge_mask = self.extract_product_advanced(product_path)
        
        # Scale if needed
        if scale != 1.0:
            new_size = (int(product_rgb.width * scale), int(product_rgb.height * scale))
            product_rgb = product_rgb.resize(new_size, Image.Resampling.LANCZOS)
            mask = mask.resize(new_size, Image.Resampling.LANCZOS)
            edge_mask = edge_mask.resize(new_size, Image.Resampling.LANCZOS)
        
        # Generate scene
        print(f"Generating scene: {scene_prompt}")
        
        if num_steps is None:
            num_steps = self.default_steps
        if cfg_weight is None:
            cfg_weight = self.default_cfg
        
        latent_h = output_size[1] // 8
        latent_w = output_size[0] // 8
        
        if seed is not None:
            mx.random.seed(seed)
        
        latents = None
        for step, x_t in enumerate(self.model.generate_latents(
            text=scene_prompt,
            n_images=1,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            latent_size=(latent_h, latent_w),
            seed=seed
        )):
            latents = x_t
            mx.eval(latents)
            if (step + 1) % max(1, num_steps // 4) == 0:
                print(f"  Step {step + 1}/{num_steps}")
        
        # Decode
        print("Decoding scene...")
        images = self.model.decode(latents)
        mx.eval(images)
        
        # Convert to PIL
        img_array = np.array(images)
        if img_array.ndim == 4:
            img_array = img_array[0]
        if img_array.shape[0] == 3:
            img_array = np.transpose(img_array, (1, 2, 0))
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        background = Image.fromarray(img_array).resize(output_size, Image.Resampling.LANCZOS)
        
        # Determine position
        if position is None:
            # Place at bottom center for better grounding
            position = (
                (background.width - product_rgb.width) // 2,
                int(background.height * 0.6)  # Lower placement
            )
        
        # Create layered shadow
        print("Creating shadows...")
        shadow_params = {
            'offset': (10, 10),
            'opacity': 0.4
        }
        shadow_layer = self.create_layered_shadow(mask, output_size, position, shadow_params)
        
        # Composite shadow
        background = Image.alpha_composite(background.convert('RGBA'), shadow_layer).convert('RGB')
        
        # Add reflection if requested
        if add_reflection:
            print("Adding reflection...")
            refl_img, refl_mask = self.add_reflection(product_rgb, mask, strength=0.2)
            refl_pos = (position[0], position[1] + product_rgb.height + 2)
            background.paste(refl_img, refl_pos, refl_mask)
        
        # Adjust lighting if requested
        if adjust_lighting:
            print("Adjusting lighting...")
            product_rgb = self.adjust_product_lighting(product_rgb, background, mask, position)
        
        # Composite with feathering
        print("Compositing product...")
        final = self.composite_with_feathering(
            product_rgb, background, mask, edge_mask, position
        )
        
        # Save
        final.save(output_path)
        print(f"✅ Saved to {output_path}")
        
        return final


def main():
    parser = argparse.ArgumentParser(description="Improved product placement V2")
    parser.add_argument("--product", required=True, help="Product image path")
    parser.add_argument("--scene", required=True, help="Scene description")
    parser.add_argument("--output", default="product_v2.png", help="Output path")
    parser.add_argument("--position", help="Position as x,y")
    parser.add_argument("--scale", type=float, default=1.0, help="Product scale")
    parser.add_argument("--reflection", action="store_true", help="Add reflection")
    parser.add_argument("--no-lighting", action="store_true", help="Disable lighting adjustment")
    parser.add_argument("--steps", type=int, help="Generation steps")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--size", default="1024,1024", help="Output size")
    
    args = parser.parse_args()
    
    # Parse position
    position = None
    if args.position:
        x, y = map(int, args.position.split(','))
        position = (x, y)
    
    # Parse size
    width, height = map(int, args.size.split(','))
    output_size = (width, height)
    
    # Run placement
    pipeline = ImprovedProductPlacement(model_type="sdxl", float16=True)
    
    pipeline.place_product_v2(
        product_path=args.product,
        scene_prompt=args.scene,
        output_path=args.output,
        position=position,
        scale=args.scale,
        add_reflection=args.reflection,
        adjust_lighting=not args.no_lighting,
        num_steps=args.steps,
        seed=args.seed,
        output_size=output_size
    )

if __name__ == "__main__":
    main()