#!/usr/bin/env python3
"""
Run V3 CorePulse on honey jar with hands scenarios
"""

import json
from corepulse_mlx_v3 import MLXCorePulseV3, MLXCorePulseV3Config
import time

def run_honey_hands():
    """Generate honey jar with hands using V3"""
    
    print("🍯 V3 Honey Jar + Hands Generation")
    print("="*60)
    
    # Load test configuration
    with open("/Users/speed/Downloads/corpus-mlx/honey_jar_test_config.json", "r") as f:
        config = json.load(f)
    
    # Initialize V3 with optimal settings
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
    
    # Process each scenario
    results = []
    reference_path = config["reference_image"]
    
    for scenario in config["test_scenarios"]:
        print(f"\n🎬 Processing: {scenario['id']}")
        print(f"   Hand Type: {scenario['hand_details']}")
        print(f"   Scene: {scenario['prompt'][:70]}...")
        
        start_time = time.time()
        
        # Generate with V3
        result = corepulse.generate_perfect_zero_hallucination_image_v3(
            prompt=scenario['prompt'],
            reference_image_path=reference_path,
            output_path=f"/Users/speed/Downloads/corpus-mlx/{scenario['output']}"
        )
        
        elapsed = time.time() - start_time
        
        # Extract metrics
        score = result.get('preservation_score', 0.0)
        
        print(f"   ✅ Generated in {elapsed:.2f}s")
        print(f"   📊 Quality Score: {score:.3f}")
        print(f"   💾 Saved: {scenario['output']}")
        
        results.append({
            "scenario": scenario['id'],
            "hand_details": scenario['hand_details'],
            "score": score,
            "time": elapsed,
            "output": scenario['output']
        })
    
    # Summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    total_score = sum(r['score'] for r in results)
    avg_score = total_score / len(results) if results else 0
    success_count = sum(1 for r in results if r['score'] > 0.85)
    success_rate = (success_count / len(results) * 100) if results else 0
    
    print(f"✅ Scenarios Completed: {len(results)}/5")
    print(f"📈 Average Score: {avg_score:.3f}")
    print(f"🎯 Success Rate: {success_rate:.0f}% ({success_count}/{len(results)})")
    
    print("\n📸 Individual Scores:")
    for r in results:
        status = "✅" if r['score'] > 0.85 else "⚠️"
        print(f"   {status} {r['scenario']:15s} [{r['hand_details']:30s}] → {r['score']:.3f}")
    
    # Save results
    results_file = "/Users/speed/Downloads/corpus-mlx/honey_v3_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "avg_score": avg_score,
            "success_rate": success_rate,
            "scenarios": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    if success_rate == 100:
        print("\n🏆 PERFECT SCORE! All scenarios above 0.85 threshold!")
        print("   V3 CorePulse achieving commercial-grade quality!")
        print("   Ready for $100k/day AGC advertising production!")
    
    return results


if __name__ == "__main__":
    results = run_honey_hands()
    print("\n🎉 V3 Honey Jar Generation Complete!")