#!/usr/bin/env python3
"""
Create comparison images showing product placement improvements
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_comparison():
    """Create side-by-side comparisons of product placements."""
    
    # Define comparisons
    comparisons = [
        {
            "title": "Watch Placement Evolution",
            "images": [
                ("test_product_watch.png", "Original Product"),
                ("watch_on_desk.png", "V1: Basic Placement"),
                ("v2_watch_marble.png", "V2: Improved Integration")
            ]
        },
        {
            "title": "Headphones Placement Evolution", 
            "images": [
                ("test_product_headphones.png", "Original Product"),
                ("headphones_gaming.png", "V1: Gaming Setup"),
                ("v2_headphones_piano.png", "V2: Piano w/ Reflection")
            ]
        }
    ]
    
    for idx, comp in enumerate(comparisons):
        print(f"Creating comparison {idx + 1}: {comp['title']}")
        
        # Load images
        images = []
        max_height = 0
        total_width = 0
        
        for img_path, label in comp['images']:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                # Resize to consistent height
                target_height = 400
                ratio = target_height / img.height
                new_width = int(img.width * ratio)
                img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
                images.append((img, label))
                max_height = max(max_height, img.height)
                total_width += img.width
            else:
                print(f"  Warning: {img_path} not found")
        
        if not images:
            continue
        
        # Create canvas
        padding = 20
        label_height = 40
        canvas_width = total_width + padding * (len(images) + 1)
        canvas_height = max_height + label_height * 2 + padding * 2
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), (240, 240, 240))
        draw = ImageDraw.Draw(canvas)
        
        # Add title
        title_y = padding
        draw.text((canvas_width // 2 - len(comp['title']) * 4, title_y), 
                 comp['title'], fill=(0, 0, 0))
        
        # Place images
        x_offset = padding
        y_offset = label_height + padding
        
        for img, label in images:
            # Add label
            label_x = x_offset + img.width // 2 - len(label) * 3
            draw.text((label_x, y_offset - 20), label, fill=(50, 50, 50))
            
            # Place image
            canvas.paste(img, (x_offset, y_offset))
            
            # Add border
            draw.rectangle(
                [(x_offset - 1, y_offset - 1), 
                 (x_offset + img.width, y_offset + img.height)],
                outline=(200, 200, 200),
                width=1
            )
            
            x_offset += img.width + padding
        
        # Save comparison
        output_name = f"comparison_{idx + 1}_{comp['title'].lower().replace(' ', '_')}.png"
        canvas.save(output_name)
        print(f"  ✅ Saved: {output_name}")
    
    # Create master comparison
    print("\nCreating master comparison grid...")
    
    # Grid of all test results
    test_images = [
        ("test_product_watch.png", "Original Watch"),
        ("test_product_headphones.png", "Original Headphones"),
        ("watch_on_desk.png", "V1: Watch on Desk"),
        ("headphones_gaming.png", "V1: Gaming Setup"),
        ("v2_watch_marble.png", "V2: Marble Table"),
        ("v2_headphones_piano.png", "V2: Piano + Reflection"),
        ("quick_test_1_watch_desk.png", "Quick: Office Desk"),
        ("quick_test_2_headphones_table.png", "Quick: Wood Table")
    ]
    
    # Create 4x2 grid
    cols = 4
    rows = 2
    img_size = 256
    padding = 10
    
    canvas_width = cols * (img_size + padding) + padding
    canvas_height = rows * (img_size + padding + 30) + padding + 40
    
    master = Image.new('RGB', (canvas_width, canvas_height), (245, 245, 245))
    draw = ImageDraw.Draw(master)
    
    # Title
    title = "Product Placement Without Hallucination - Test Results"
    draw.text((canvas_width // 2 - len(title) * 4, 10), title, fill=(0, 0, 0))
    
    # Place images
    for idx, (img_path, label) in enumerate(test_images):
        if not os.path.exists(img_path):
            continue
        
        row = idx // cols
        col = idx % cols
        
        x = padding + col * (img_size + padding)
        y = 50 + row * (img_size + padding + 30)
        
        # Load and resize
        img = Image.open(img_path)
        img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
        
        # Center in cell
        img_x = x + (img_size - img.width) // 2
        img_y = y + (img_size - img.height) // 2
        
        # Paste image
        master.paste(img, (img_x, img_y))
        
        # Add border
        draw.rectangle(
            [(img_x - 1, img_y - 1), 
             (img_x + img.width, img_y + img.height)],
            outline=(180, 180, 180),
            width=1
        )
        
        # Add label
        label_text = label[:20] + "..." if len(label) > 20 else label
        label_x = x + (img_size - len(label_text) * 6) // 2
        draw.text((label_x, y + img_size + 5), label_text, fill=(60, 60, 60))
    
    master.save("master_comparison_grid.png")
    print("✅ Saved: master_comparison_grid.png")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print("Generated comparison images showing:")
    print("  1. Original products preserved perfectly")
    print("  2. V1 implementation with basic placement")
    print("  3. V2 improvements with better integration")
    print("  - Layered shadows for realism")
    print("  - Lighting adjustment to match scenes")
    print("  - Optional reflections for glossy surfaces")
    print("  - Better edge blending")

if __name__ == "__main__":
    create_comparison()