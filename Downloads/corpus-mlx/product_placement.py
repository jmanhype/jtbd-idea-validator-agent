#!/usr/bin/env python3
"""
High-Quality Product Placement without Hallucination
Uses inpainting and careful masking to preserve product integrity
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage
import cv2

from stable_diffusion import StableDiffusion, StableDiffusionXL


class ProductPlacementPipeline:
    """
    Pipeline for accurate product placement in scenes without hallucination.
    
    Key techniques:
    1. Product masking - Preserve exact product pixels
    2. Inpainting - Generate only background/context
    3. Edge blending - Smooth integration
    4. Multi-stage refinement - Progressive quality improvement
    """
    
    def __init__(
        self,
        model_type: str = "sdxl",
        model_name: Optional[str] = None,
        float16: bool = True
    ):
        """Initialize the product placement pipeline."""
        
        if model_type == "sdxl":
            self.model = StableDiffusionXL(
                model=model_name or "stabilityai/sdxl-turbo",
                float16=float16
            )
            self.is_sdxl = True
            self.default_steps = 4
            self.default_cfg = 0.0
        else:
            self.model = StableDiffusion(
                model=model_name or "stabilityai/stable-diffusion-2-1-base",
                float16=float16
            )
            self.is_sdxl = False
            self.default_steps = 30
            self.default_cfg = 7.5
        
        print(f"Loading {model_type} model...")
        self.model.ensure_models_are_loaded()
        print("Model loaded successfully!")
    
    def create_product_mask(
        self,
        image: Image.Image,
        method: str = "auto",
        threshold: int = 240,
        expand_pixels: int = 0
    ) -> Image.Image:
        """
        Create a mask for the product in the image.
        
        Args:
            image: Input product image
            method: 'auto' for automatic, 'manual' for user-provided mask
            threshold: Threshold for background removal (for white backgrounds)
            expand_pixels: Expand mask by N pixels for safety margin
        
        Returns:
            Binary mask (white = product, black = background)
        """
        
        if method == "auto":
            # Convert to numpy array
            img_array = np.array(image.convert("RGBA"))
            
            # If image has alpha channel, use it
            if img_array.shape[2] == 4 and not np.all(img_array[:, :, 3] == 255):
                mask = img_array[:, :, 3]
                mask = (mask > 128).astype(np.uint8) * 255
            else:
                # Assume white/light background
                gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
                
                # Threshold to separate product from background
                _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
                
                # Clean up mask with morphological operations
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Expand mask if requested
            if expand_pixels > 0:
                kernel = np.ones((expand_pixels*2+1, expand_pixels*2+1), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
            
            return Image.fromarray(mask, mode='L')
        
        else:
            raise NotImplementedError("Manual mask method not yet implemented")
    
    def create_inpainting_mask(
        self,
        product_mask: Image.Image,
        blur_radius: int = 10,
        expand_pixels: int = 20
    ) -> Image.Image:
        """
        Create an inpainting mask with soft edges for smooth blending.
        
        Args:
            product_mask: Binary product mask
            blur_radius: Radius for edge softening
            expand_pixels: Expand the inpainting area
        
        Returns:
            Soft inpainting mask
        """
        
        mask = product_mask.convert('L')
        
        # Invert mask (we want to inpaint the background, not the product)
        mask = ImageOps.invert(mask)
        
        # Expand the inpainting region slightly
        if expand_pixels > 0:
            mask_array = np.array(mask)
            kernel = np.ones((expand_pixels*2+1, expand_pixels*2+1), np.uint8)
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
            mask = Image.fromarray(mask_array, mode='L')
        
        # Apply Gaussian blur for soft edges
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return mask
    
    def composite_product(
        self,
        product_image: Image.Image,
        background: Image.Image,
        product_mask: Image.Image,
        position: Optional[Tuple[int, int]] = None,
        scale: float = 1.0,
        shadow: bool = True,
        shadow_opacity: float = 0.3,
        shadow_blur: int = 10,
        shadow_offset: Tuple[int, int] = (5, 5)
    ) -> Image.Image:
        """
        Composite product onto background with optional shadow.
        
        Args:
            product_image: The product image
            background: The generated background
            product_mask: Product mask
            position: Where to place product (None = center)
            scale: Scale factor for product
            shadow: Whether to add drop shadow
            shadow_opacity: Shadow darkness (0-1)
            shadow_blur: Shadow blur radius
            shadow_offset: Shadow offset (x, y)
        
        Returns:
            Final composited image
        """
        
        # Resize product if needed
        if scale != 1.0:
            new_size = (
                int(product_image.width * scale),
                int(product_image.height * scale)
            )
            product_image = product_image.resize(new_size, Image.Resampling.LANCZOS)
            product_mask = product_mask.resize(new_size, Image.Resampling.LANCZOS)
        
        # Determine position
        if position is None:
            # Center the product
            position = (
                (background.width - product_image.width) // 2,
                (background.height - product_image.height) // 2
            )
        
        # Create a copy of the background
        result = background.copy()
        
        # Add shadow if requested
        if shadow:
            # Create shadow from mask
            shadow_img = Image.new('RGBA', background.size, (0, 0, 0, 0))
            shadow_mask = product_mask.copy()
            
            # Apply blur to shadow
            shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
            
            # Position shadow with offset
            shadow_pos = (position[0] + shadow_offset[0], position[1] + shadow_offset[1])
            
            # Create a properly sized shadow at the correct position
            temp_shadow = Image.new('L', background.size, 0)
            temp_shadow.paste(shadow_mask, shadow_pos)
            
            # Create shadow layer with opacity
            shadow_layer = Image.new('RGBA', background.size, (0, 0, 0, 0))
            shadow_layer.paste((0, 0, 0, int(255 * shadow_opacity)), mask=temp_shadow)
            
            # Composite shadow onto result
            result = Image.alpha_composite(result.convert('RGBA'), shadow_layer).convert('RGB')
        
        # Paste the product using its mask
        result.paste(product_image, position, product_mask)
        
        return result
    
    def place_product(
        self,
        product_path: Union[str, Path],
        scene_prompt: str,
        output_path: Union[str, Path],
        product_description: Optional[str] = None,
        negative_prompt: str = "",
        position: Optional[Tuple[int, int]] = None,
        scale: float = 1.0,
        num_steps: Optional[int] = None,
        cfg_weight: Optional[float] = None,
        seed: Optional[int] = None,
        mask_threshold: int = 240,
        add_shadow: bool = True,
        preserve_product: bool = True,
        output_size: Tuple[int, int] = (1024, 1024)
    ) -> Image.Image:
        """
        Main method to place a product in a generated scene.
        
        Args:
            product_path: Path to product image
            scene_prompt: Description of the scene/background
            output_path: Where to save the result
            product_description: Description of the product for context
            negative_prompt: What to avoid in generation
            position: Where to place product (None = center)
            scale: Scale factor for product
            num_steps: Number of denoising steps
            cfg_weight: Classifier-free guidance weight
            seed: Random seed for reproducibility
            mask_threshold: Threshold for auto-masking
            add_shadow: Whether to add drop shadow
            preserve_product: If True, use original product pixels (no hallucination)
            output_size: Output image size
        
        Returns:
            Final composed image
        """
        
        # Load product image
        product_image = Image.open(product_path).convert("RGBA")
        print(f"Loaded product image: {product_image.size}")
        
        # Create product mask
        print("Creating product mask...")
        product_mask = self.create_product_mask(
            product_image,
            method="auto",
            threshold=mask_threshold
        )
        
        # Prepare the prompt
        if product_description:
            full_prompt = f"{scene_prompt}, with {product_description} in the scene"
        else:
            full_prompt = scene_prompt
        
        print(f"Generating scene: {full_prompt}")
        
        # Set generation parameters
        if num_steps is None:
            num_steps = self.default_steps
        if cfg_weight is None:
            cfg_weight = self.default_cfg
        
        # Calculate latent size
        if self.is_sdxl:
            latent_h = output_size[1] // 8
            latent_w = output_size[0] // 8
        else:
            latent_h = output_size[1] // 8
            latent_w = output_size[0] // 8
        
        # Generate the background scene
        print(f"Generating background (steps={num_steps}, cfg={cfg_weight})...")
        
        if seed is not None:
            mx.random.seed(seed)
        
        # Generate latents
        latents = None
        for step, x_t in enumerate(self.model.generate_latents(
            text=full_prompt,
            n_images=1,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            negative_text=negative_prompt,
            latent_size=(latent_h, latent_w),
            seed=seed
        )):
            latents = x_t
            mx.eval(latents)
            if (step + 1) % max(1, num_steps // 4) == 0:
                print(f"  Step {step + 1}/{num_steps}")
        
        # Decode the latents
        print("Decoding generated scene...")
        background = self.model.decode(latents)
        mx.eval(background)
        
        # Convert to PIL Image
        bg_array = np.array(background)
        if bg_array.ndim == 4:
            bg_array = bg_array[0]
        if bg_array.shape[0] == 3:
            bg_array = np.transpose(bg_array, (1, 2, 0))
        bg_array = np.clip(bg_array * 255, 0, 255).astype(np.uint8)
        background_img = Image.fromarray(bg_array).resize(output_size, Image.Resampling.LANCZOS)
        
        if preserve_product:
            # Composite the original product onto the generated background
            print("Compositing product onto scene...")
            
            # Convert product to RGB if needed
            if product_image.mode == "RGBA":
                product_rgb = Image.new("RGB", product_image.size, (255, 255, 255))
                product_rgb.paste(product_image, mask=product_image.split()[3])
            else:
                product_rgb = product_image.convert("RGB")
            
            # Composite
            final_image = self.composite_product(
                product_image=product_rgb,
                background=background_img,
                product_mask=product_mask,
                position=position,
                scale=scale,
                shadow=add_shadow
            )
        else:
            # Use the generated image as-is (may have hallucinations)
            final_image = background_img
        
        # Save the result
        final_image.save(output_path)
        print(f"✅ Saved result to: {output_path}")
        
        return final_image


def main():
    parser = argparse.ArgumentParser(
        description="High-quality product placement without hallucination"
    )
    parser.add_argument(
        "--product",
        type=str,
        required=True,
        help="Path to product image"
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene description prompt"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="product_placement_result.png",
        help="Output image path"
    )
    parser.add_argument(
        "--product-desc",
        type=str,
        help="Description of the product for context"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sd", "sdxl"],
        default="sdxl",
        help="Model type to use"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for product (0.1-2.0)"
    )
    parser.add_argument(
        "--position",
        type=str,
        help="Position as 'x,y' coordinates (default: center)"
    )
    parser.add_argument(
        "--no-shadow",
        action="store_true",
        help="Disable drop shadow"
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1024,1024",
        help="Output size as 'width,height'"
    )
    
    args = parser.parse_args()
    
    # Parse position if provided
    position = None
    if args.position:
        x, y = map(int, args.position.split(','))
        position = (x, y)
    
    # Parse size
    width, height = map(int, args.size.split(','))
    output_size = (width, height)
    
    # Initialize pipeline
    pipeline = ProductPlacementPipeline(
        model_type=args.model,
        float16=True
    )
    
    # Place the product
    result = pipeline.place_product(
        product_path=args.product,
        scene_prompt=args.scene,
        output_path=args.output,
        product_description=args.product_desc,
        position=position,
        scale=args.scale,
        num_steps=args.steps,
        seed=args.seed,
        add_shadow=not args.no_shadow,
        preserve_product=True,  # Always preserve original product
        output_size=output_size
    )
    
    print(f"\n✨ Product placement complete!")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()