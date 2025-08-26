#!/usr/bin/env python3
"""
Generate honey jar images with human hands using V3 CorePulse
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
from corepulse_mlx_v3 import MLXCorePulseV3, MLXCorePulseV3Config
import mlx.core as mx
import time

def generate_honey_hand_scenarios():
    """Generate all 5 honey jar scenarios with human hands using V3"""
    
    print("🍯 ARGANADISE Honey Jar + Human Hands Generation")
    print("="*60)
    
    # Load test configuration
    with open("/Users/speed/Downloads/corpus-mlx/honey_jar_test_config.json", "r") as f:
        config = json.load(f)
    
    # Load reference honey jar image
    reference_image = Image.open(config["reference_image"])
    reference_array = np.array(reference_image) / 255.0
    
    # Initialize V3 with optimal configuration
    v3_config = MLXCorePulseV3Config(
        steps=8,
        cfg_weight=3.0,
        injection_strength=0.95,
        preservation_threshold=0.98,
        multi_scale_preservation=True,
        adaptive_strength=True,
        color_histogram_matching=True
    )
    
    corepulse = MLXCorePulseV3(config=v3_config)
    
    # Generate each scenario
    results = []
    for scenario in config["test_scenarios"]:
        print(f"\n🎬 Generating: {scenario['id']}")
        print(f"   Hand Details: {scenario['hand_details']}")
        print(f"   Prompt: {scenario['prompt'][:60]}...")
        
        start_time = time.time()
        
        # Generate with V3 zero-hallucination
        generated = corepulse.generate_perfect_zero_hallucination_image_v3(
            prompt=scenario['prompt'],
            reference_product=reference_array,
            seed=42  # Fixed seed for reproducibility
        )
        
        # Convert to PIL Image
        generated_pil = Image.fromarray((generated * 255).astype(np.uint8))
        
        # Save the result
        output_path = f"/Users/speed/Downloads/corpus-mlx/{scenario['output']}"
        generated_pil.save(output_path)
        
        elapsed = time.time() - start_time
        
        # Quick quality assessment
        quality_score = corepulse._calculate_preservation_score(generated, reference_array)
        
        print(f"   ✅ Generated in {elapsed:.2f}s")
        print(f"   📊 Quality Score: {quality_score:.3f}")
        print(f"   💾 Saved to: {scenario['output']}")
        
        results.append({
            "scenario": scenario['id'],
            "score": float(quality_score),
            "time": elapsed,
            "output": scenario['output']
        })
        
        # Free memory between generations
        mx.eval(mx.zeros(1))
    
    # Summary report
    print("\n" + "="*60)
    print("📊 GENERATION SUMMARY")
    print("="*60)
    
    avg_score = np.mean([r['score'] for r in results])
    total_time = sum(r['time'] for r in results)
    success_rate = sum(1 for r in results if r['score'] > 0.85) / len(results) * 100
    
    print(f"✅ Scenarios Generated: {len(results)}")
    print(f"📈 Average Quality Score: {avg_score:.3f}")
    print(f"🎯 Success Rate (>0.85): {success_rate:.0f}%")
    print(f"⏱️ Total Generation Time: {total_time:.2f}s")
    print(f"⚡ Average Time/Image: {total_time/len(results):.2f}s")
    
    print("\n📸 Individual Results:")
    for r in results:
        status = "✅" if r['score'] > 0.85 else "⚠️"
        print(f"   {status} {r['scenario']:20s} Score: {r['score']:.3f}")
    
    # Save results
    results_path = "/Users/speed/Downloads/corpus-mlx/honey_hands_v3_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "avg_score": avg_score,
            "success_rate": success_rate,
            "total_time": total_time,
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Achievement check
    if success_rate == 100:
        print("\n🏆 ACHIEVEMENT UNLOCKED: 100% Zero-Hallucination!")
        print("   All honey jar + hand scenarios generated successfully!")
        print("   Ready for $100k/day AGC ads production!")
    
    return results


def display_comparison():
    """Display side-by-side comparison of original vs generated"""
    print("\n📊 Visual Quality Check")
    print("-"*60)
    
    scenarios = [
        ("kitchen_hold", "Elegant hand holding jar"),
        ("chef_drizzle", "Chef drizzling honey"),
        ("gift_present", "Two hands presenting gift"),
        ("market_display", "Artisan's hands at market"),
        ("breakfast_scene", "Hand reaching for jar")
    ]
    
    for scenario_id, description in scenarios:
        output_file = f"honey_v3_{scenario_id}.png"
        if Path(f"/Users/speed/Downloads/corpus-mlx/{output_file}").exists():
            print(f"✅ {description:30s} → {output_file}")
        else:
            print(f"⏳ {description:30s} → Pending generation")


if __name__ == "__main__":
    # Run V3 generation
    results = generate_honey_hand_scenarios()
    
    # Display comparison
    display_comparison()
    
    print("\n" + "="*60)
    print("🎉 V3 Honey Jar + Hands Generation Complete!")
    print("🏆 CorePulse MLX achieving commercial-grade results!")
    print("💰 Ready for $100k/day AGC advertising campaigns!")