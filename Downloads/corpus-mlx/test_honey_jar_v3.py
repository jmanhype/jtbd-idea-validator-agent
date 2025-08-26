#!/usr/bin/env python3
"""
Test V3 CorePulse with Honey Jar Product Placement
Including human hand holding for realistic commercial photography
"""

import sys
import numpy as np
from PIL import Image
from pathlib import Path

# Import our V3 implementation
from corepulse_mlx_v3 import MLXCorePulseV3

def test_honey_jar_placement():
    """Test V3 with honey jar including person holding it"""
    
    print("🍯 Testing V3 CorePulse with Honey Jar Product Placement")
    print("=" * 60)
    
    # Initialize V3 with config
    from corepulse_mlx_v3 import MLXCorePulseV3Config
    config = MLXCorePulseV3Config()
    corepulse_v3 = MLXCorePulseV3(config)
    
    # Test scenarios with human interaction
    test_scenarios = [
        {
            "prompt": "elegant hand holding premium honey jar in bright kitchen, morning sunlight streaming through window, marble countertop, professional food photography",
            "output": "honey_jar_hand_kitchen.png",
            "description": "Hand holding honey jar in luxury kitchen"
        },
        {
            "prompt": "woman's hand presenting artisanal honey jar at farmers market stall, rustic wooden table, warm natural lighting, lifestyle photography",
            "output": "honey_jar_hand_market.png", 
            "description": "Hand presenting honey at farmers market"
        },
        {
            "prompt": "chef's hand drizzling honey from premium jar onto gourmet breakfast plate, restaurant setting, professional culinary photography, shallow depth of field",
            "output": "honey_jar_chef_action.png",
            "description": "Chef using honey jar in action shot"
        },
        {
            "prompt": "hands holding honey jar gift with ribbon, cozy living room background, warm afternoon light, lifestyle product photography",
            "output": "honey_jar_gift_hands.png",
            "description": "Hands presenting honey jar as gift"
        }
    ]
    
    # Reference image path (user's honey jar)
    reference_image = "/Users/speed/Downloads/corpus-mlx/honey_jar_reference.png"
    
    # Save the user's image first if needed
    print("📸 Using honey jar reference image...")
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎯 Scenario {i}/{len(test_scenarios)}: {scenario['description']}")
        print(f"📝 Prompt: {scenario['prompt'][:80]}...")
        
        try:
            # Generate with V3 perfect zero-hallucination
            result = corepulse_v3.generate_perfect_zero_hallucination_image_v3(
                prompt=scenario['prompt'],
                reference_image_path=reference_image,
                output_path=f"/Users/speed/Downloads/corpus-mlx/{scenario['output']}"
            )
            
            print(f"✅ Generated: {scenario['output']}")
            print(f"   Score: {result['quality_score']:.3f}")
            print(f"   Success: {result['perfect_success']}")
            
            results.append({
                "scenario": scenario['description'],
                "output": scenario['output'],
                "score": result['quality_score'],
                "success": result['perfect_success']
            })
            
        except Exception as e:
            print(f"❌ Error generating scenario {i}: {str(e)}")
            continue
    
    # Summary report
    print("\n" + "=" * 60)
    print("📊 HONEY JAR PLACEMENT RESULTS")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r['success'])
    success_rate = success_count / len(results) if results else 0
    
    print(f"Success Rate: {success_rate:.1%} ({success_count}/{len(results)})")
    print(f"Target: 100% zero-hallucination with human interaction")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['scenario']}: {result['score']:.3f}")
    
    if success_rate == 1.0:
        print("\n🏆 PERFECT! All honey jar placements with human hands successful!")
        print("🍯 Ready for commercial product photography with human interaction!")
    
    return results


if __name__ == "__main__":
    # First, save the reference image from user input
    print("💾 Please ensure the honey jar image is saved as 'honey_jar_reference.png'")
    print("   in /Users/speed/Downloads/corpus-mlx/")
    
    # Check if reference exists or create placeholder
    reference_path = Path("/Users/speed/Downloads/corpus-mlx/honey_jar_reference.png")
    if not reference_path.exists():
        print("⚠️  Reference image not found. Please save the honey jar image first.")
        print("   Expected path: /Users/speed/Downloads/corpus-mlx/honey_jar_reference.png")
        sys.exit(1)
    
    # Run the test
    test_honey_jar_placement()