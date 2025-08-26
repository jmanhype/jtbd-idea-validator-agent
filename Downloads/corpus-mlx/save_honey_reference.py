#!/usr/bin/env python3
"""
Save honey jar reference image for V3 testing
Note: Since the image was provided by user, we'll create a test reference
"""

import numpy as np
from PIL import Image
import requests
from io import BytesIO

def create_honey_jar_reference():
    """Create a reference honey jar image for testing"""
    
    print("🍯 Creating honey jar reference for V3 testing...")
    
    # Since we can't directly access the user's uploaded image,
    # we'll create a high-quality placeholder or use a similar reference
    # In production, the user would provide the actual image file
    
    # Create a placeholder that represents the honey jar characteristics
    # Dimensions based on typical product photo
    width, height = 512, 512
    
    # Create base image with white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 245  # Light gray/white
    
    # Add a central region for the jar (simplified representation)
    center_x, center_y = width // 2, height // 2
    jar_width, jar_height = 200, 280
    
    # Jar region (amber/golden color similar to honey)
    x1 = center_x - jar_width // 2
    x2 = center_x + jar_width // 2
    y1 = center_y - jar_height // 2 + 30
    y2 = center_y + jar_height // 2 - 30
    
    # Golden amber color for jar body
    image[y1:y2, x1:x2] = [218, 165, 32]  # Goldenrod color
    
    # Add lid region (gold metallic)
    lid_height = 40
    image[y1-lid_height:y1, x1:x2] = [255, 215, 0]  # Gold color
    
    # Add label region (darker band)
    label_y1 = y1 + 60
    label_y2 = label_y1 + 80
    image[label_y1:label_y2, x1:x2] = [139, 69, 19]  # Saddle brown
    
    # Add reflection at bottom (lighter region)
    reflection_height = 30
    for i in range(reflection_height):
        alpha = 1.0 - (i / reflection_height) * 0.5
        image[y2+i:y2+i+1, x1:x2] = image[y2+i:y2+i+1, x1:x2] * alpha + np.array([255, 255, 255]) * (1-alpha)
    
    # Save the reference image
    pil_image = Image.fromarray(image.astype(np.uint8))
    output_path = "/Users/speed/Downloads/corpus-mlx/honey_jar_reference.png"
    pil_image.save(output_path)
    
    print(f"✅ Saved honey jar reference to: {output_path}")
    print("   Note: This is a simplified reference. For production, use actual product photo.")
    
    return output_path


if __name__ == "__main__":
    create_honey_jar_reference()