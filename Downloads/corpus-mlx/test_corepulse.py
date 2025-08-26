#!/usr/bin/env python3
"""
Test CorePulse-MLX capabilities
Demonstrates advanced control over diffusion process
"""

import numpy as np
from PIL import Image
import mlx.core as mx
from stable_diffusion import StableDiffusionXL
from corepulse_mlx import (
    CorePulseMLX, 
    CorePulsePresets,
    PromptInjection,
    InjectionLevel,
    TokenMask,
    SpatialInjection
)


def test_style_content_separation():
    """Test separating style and content control"""
    print("\n" + "="*60)
    print("TEST 1: Style/Content Separation")
    print("="*60)
    
    # Initialize
    base_model = StableDiffusionXL(
        model="stabilityai/sdxl-turbo",
        float16=True
    )
    base_model.ensure_models_are_loaded()
    
    corepulse = CorePulseMLX(base_model)
    
    # Define separate style and content
    content = "a vintage motorcycle"
    style = "cyberpunk neon art style, glowing edges, dark background"
    
    injections = CorePulsePresets.style_content_separation(
        content_prompt=content,
        style_prompt=style,
        strength=0.8
    )
    
    print(f"Content: {content}")
    print(f"Style: {style}")
    print("Injecting content in early blocks, style in late blocks...")
    
    # Generate
    result = corepulse.generate_with_control(
        base_prompt=f"{content}, {style}",
        prompt_injections=injections,
        num_steps=4,
        cfg_weight=0.0,
        seed=42,
        output_size=(1024, 1024)
    )
    
    # Convert and save
    img_array = np.array(result)
    if img_array.ndim == 4:
        img_array = img_array[0]
    if img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save("corepulse_style_content.png")
    print("✅ Saved: corepulse_style_content.png")


def test_progressive_detail():
    """Test progressive detail injection"""
    print("\n" + "="*60)
    print("TEST 2: Progressive Detail Control")
    print("="*60)
    
    # Initialize
    base_model = StableDiffusionXL(
        model="stabilityai/sdxl-turbo",
        float16=True
    )
    base_model.ensure_models_are_loaded()
    
    corepulse = CorePulseMLX(base_model)
    
    # Define progressive prompts
    structure = "architectural building, modern design"
    details = "glass windows, steel beams, concrete"
    fine_details = "reflections, shadows, ambient lighting, people walking"
    
    injections = CorePulsePresets.progressive_detail(
        structure_prompt=structure,
        detail_prompt=details,
        fine_detail_prompt=fine_details
    )
    
    print(f"Structure: {structure}")
    print(f"Details: {details}")
    print(f"Fine details: {fine_details}")
    print("Injecting progressively across UNet blocks...")
    
    # Generate
    result = corepulse.generate_with_control(
        base_prompt=f"{structure}, {details}, {fine_details}",
        prompt_injections=injections,
        num_steps=4,
        cfg_weight=0.0,
        seed=123,
        output_size=(1024, 1024)
    )
    
    # Convert and save
    img_array = np.array(result)
    if img_array.ndim == 4:
        img_array = img_array[0]
    if img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save("corepulse_progressive_detail.png")
    print("✅ Saved: corepulse_progressive_detail.png")


def test_token_emphasis():
    """Test token-level attention control"""
    print("\n" + "="*60)
    print("TEST 3: Token-Level Attention Control")
    print("="*60)
    
    # Initialize
    base_model = StableDiffusionXL(
        model="stabilityai/sdxl-turbo",
        float16=True
    )
    base_model.ensure_models_are_loaded()
    
    corepulse = CorePulseMLX(base_model)
    
    prompt = "a red ferrari and blue lamborghini racing on track"
    
    # Emphasize "ferrari" and suppress "lamborghini"
    token_masks = [
        TokenMask(
            tokens=["ferrari"],
            mask_type="amplify",
            strength=2.0
        ),
        TokenMask(
            tokens=["lamborghini"],
            mask_type="suppress",
            strength=0.7
        )
    ]
    
    print(f"Prompt: {prompt}")
    print("Amplifying 'ferrari' (2x), Suppressing 'lamborghini' (0.3x)")
    
    # Generate
    result = corepulse.generate_with_control(
        base_prompt=prompt,
        token_masks=token_masks,
        num_steps=4,
        cfg_weight=0.0,
        seed=456,
        output_size=(1024, 1024)
    )
    
    # Convert and save
    img_array = np.array(result)
    if img_array.ndim == 4:
        img_array = img_array[0]
    if img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save("corepulse_token_control.png")
    print("✅ Saved: corepulse_token_control.png")


def test_spatial_injection():
    """Test spatial region control"""
    print("\n" + "="*60)
    print("TEST 4: Spatial Region Control")
    print("="*60)
    
    # Initialize
    base_model = StableDiffusionXL(
        model="stabilityai/sdxl-turbo",
        float16=True
    )
    base_model.ensure_models_are_loaded()
    
    corepulse = CorePulseMLX(base_model)
    
    base_prompt = "a landscape with mountains and lake"
    
    # Different prompts for different regions
    spatial_injections = [
        SpatialInjection(
            prompt="dramatic sunset sky, orange and purple colors",
            bbox=(0, 0, 1024, 400),  # Top region (sky)
            strength=0.8,
            feather=30
        ),
        SpatialInjection(
            prompt="crystal clear water, reflections",
            bbox=(0, 600, 1024, 1024),  # Bottom region (lake)
            strength=0.7,
            feather=20
        )
    ]
    
    print(f"Base: {base_prompt}")
    print("Injecting 'sunset sky' in top region")
    print("Injecting 'crystal water' in bottom region")
    
    # Generate
    result = corepulse.generate_with_control(
        base_prompt=base_prompt,
        spatial_injections=spatial_injections,
        num_steps=4,
        cfg_weight=0.0,
        seed=789,
        output_size=(1024, 1024)
    )
    
    # Convert and save
    img_array = np.array(result)
    if img_array.ndim == 4:
        img_array = img_array[0]
    if img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save("corepulse_spatial_control.png")
    print("✅ Saved: corepulse_spatial_control.png")


def test_combined_control():
    """Test combining multiple control methods"""
    print("\n" + "="*60)
    print("TEST 5: Combined Multi-Level Control")
    print("="*60)
    
    # Initialize
    base_model = StableDiffusionXL(
        model="stabilityai/sdxl-turbo",
        float16=True
    )
    base_model.ensure_models_are_loaded()
    
    corepulse = CorePulseMLX(base_model)
    
    # Base scene
    base_prompt = "futuristic city street with flying cars"
    
    # Style/content injection
    injections = [
        PromptInjection(
            prompt="cyberpunk architecture, neon signs",
            levels=[InjectionLevel.ENCODER_MID],
            strength=0.8
        ),
        PromptInjection(
            prompt="blade runner atmosphere, rain, reflections",
            levels=[InjectionLevel.DECODER_LATE],
            strength=0.7
        )
    ]
    
    # Token emphasis
    token_masks = [
        TokenMask(
            tokens=["neon"],
            mask_type="amplify",
            strength=1.5
        )
    ]
    
    # Spatial control
    spatial_injections = [
        SpatialInjection(
            prompt="bright holographic advertisements",
            bbox=(0, 200, 400, 600),  # Left side
            strength=0.6,
            feather=25
        )
    ]
    
    print(f"Base: {base_prompt}")
    print("Applying multiple control layers:")
    print("  - Style injection: cyberpunk + blade runner")
    print("  - Token emphasis: amplify 'neon'")
    print("  - Spatial: holographic ads on left")
    
    # Generate
    result = corepulse.generate_with_control(
        base_prompt=base_prompt,
        prompt_injections=injections,
        token_masks=token_masks,
        spatial_injections=spatial_injections,
        num_steps=4,
        cfg_weight=0.0,
        seed=999,
        output_size=(1024, 1024)
    )
    
    # Convert and save
    img_array = np.array(result)
    if img_array.ndim == 4:
        img_array = img_array[0]
    if img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save("corepulse_combined_control.png")
    print("✅ Saved: corepulse_combined_control.png")


def main():
    print("\n" + "="*70)
    print("COREPULSE-MLX TEST SUITE")
    print("Advanced Diffusion Control for MLX")
    print("="*70)
    
    tests = [
        ("Style/Content Separation", test_style_content_separation),
        ("Progressive Detail Control", test_progressive_detail),
        ("Token-Level Attention", test_token_emphasis),
        ("Spatial Region Control", test_spatial_injection),
        ("Combined Multi-Level Control", test_combined_control)
    ]
    
    print(f"\nRunning {len(tests)} tests...")
    
    for i, (name, test_func) in enumerate(tests, 1):
        try:
            test_func()
            print(f"✅ Test {i}/{len(tests)} passed: {name}")
        except Exception as e:
            print(f"❌ Test {i}/{len(tests)} failed: {name}")
            print(f"   Error: {e}")
    
    print("\n" + "="*70)
    print("COREPULSE TEST COMPLETE")
    print("="*70)
    print("\nGenerated images demonstrate:")
    print("  • Style and content separation")
    print("  • Progressive detail injection")
    print("  • Token-level attention control")
    print("  • Spatial region manipulation")
    print("  • Combined multi-level control")
    print("\nThis brings CorePulse-style capabilities to MLX!")


if __name__ == "__main__":
    main()