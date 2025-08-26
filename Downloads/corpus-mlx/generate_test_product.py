#!/usr/bin/env python3
"""
Generate a test product image for product placement testing
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mlx.core as mx
from stable_diffusion import StableDiffusionXL

def generate_product_image():
    """Generate a simple product image with transparent background."""
    
    print("Generating test product with SDXL...")
    
    # Initialize SDXL
    sdxl = StableDiffusionXL(
        model="stabilityai/sdxl-turbo",
        float16=True
    )
    sdxl.ensure_models_are_loaded()
    
    # Generate a product on white background
    prompt = "professional product photo of a red wireless headphones, centered, white background, studio lighting, high quality, product photography"
    negative_prompt = "shadows, reflections, multiple objects, text, watermark"
    
    print(f"Generating: {prompt}")
    
    # Generate with minimal steps for speed
    latents = None
    for x_t in sdxl.generate_latents(
        text=prompt,
        n_images=1,
        num_steps=4,
        cfg_weight=0.0,
        latent_size=(128, 128),
        seed=42
    ):
        latents = x_t
        mx.eval(latents)
    
    # Decode
    images = sdxl.decode(latents)
    mx.eval(images)
    
    # Convert to PIL
    img_array = np.array(images)
    if img_array.ndim == 4:
        img_array = img_array[0]
    if img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    
    product_img = Image.fromarray(img_array)
    product_img.save("test_product_headphones.png")
    print("✅ Saved test_product_headphones.png")
    
    # Also generate a simple geometric product for easier masking
    print("\nGenerating geometric product...")
    
    # Create a simple product image (a watch)
    img = Image.new('RGBA', (512, 512), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a watch shape
    # Watch face
    center = (256, 256)
    radius = 150
    draw.ellipse([center[0]-radius, center[1]-radius, 
                  center[0]+radius, center[1]+radius], 
                 fill=(50, 50, 50, 255), outline=(30, 30, 30, 255), width=5)
    
    # Inner circle
    inner_radius = 140
    draw.ellipse([center[0]-inner_radius, center[1]-inner_radius,
                  center[0]+inner_radius, center[1]+inner_radius],
                 fill=(240, 240, 240, 255))
    
    # Watch hands
    # Hour hand
    draw.line([center, (center[0], center[1]-80)], fill=(0, 0, 0, 255), width=6)
    # Minute hand
    draw.line([center, (center[0]+60, center[1]-60)], fill=(0, 0, 0, 255), width=4)
    # Second hand
    draw.line([center, (center[0]-30, center[1]+90)], fill=(255, 0, 0, 255), width=2)
    
    # Center dot
    draw.ellipse([center[0]-8, center[1]-8, center[0]+8, center[1]+8],
                 fill=(0, 0, 0, 255))
    
    # Watch band (top)
    draw.rectangle([center[0]-40, center[1]-radius-50, 
                    center[0]+40, center[1]-radius],
                   fill=(139, 69, 19, 255))
    
    # Watch band (bottom)
    draw.rectangle([center[0]-40, center[1]+radius,
                    center[0]+40, center[1]+radius+50],
                   fill=(139, 69, 19, 255))
    
    # Add some details - hour markers
    for i in range(12):
        angle = i * 30 * np.pi / 180
        x1 = center[0] + int(120 * np.sin(angle))
        y1 = center[1] - int(120 * np.cos(angle))
        x2 = center[0] + int(130 * np.sin(angle))
        y2 = center[1] - int(130 * np.cos(angle))
        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0, 255), width=2)
    
    img.save("test_product_watch.png")
    print("✅ Saved test_product_watch.png")
    
    return True

if __name__ == "__main__":
    generate_product_image()