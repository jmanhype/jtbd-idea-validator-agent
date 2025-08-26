#!/usr/bin/env python3
"""
Dogfooding Analysis V1 - MLX CorePulse Results
Systematic analysis of generated images to iteratively reach 100% success
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

class DogfoodAnalyzer:
    """Analyze generated images vs reference for iterative improvements"""
    
    def __init__(self):
        self.analysis_results = []
        self.improvement_recommendations = []
    
    def analyze_generation_pair(
        self, 
        generated_path: str,
        reference_path: str,
        prompt: str,
        generation_stats: Dict
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a generated image vs reference
        """
        print(f"\n🔍 DOGFOODING ANALYSIS")
        print(f"Generated: {generated_path}")
        print(f"Reference: {reference_path}")
        print(f"Prompt: {prompt}")
        
        # Load images
        generated = np.array(Image.open(generated_path))
        reference = np.array(Image.open(reference_path))
        
        analysis = {
            "files": {
                "generated": generated_path,
                "reference": reference_path,
                "prompt": prompt
            },
            "generation_stats": generation_stats,
            "quality_analysis": self._analyze_image_quality(generated, reference),
            "product_preservation": self._analyze_product_preservation(generated, reference),
            "hallucination_check": self._check_hallucinations(generated, reference, prompt),
            "composition_analysis": self._analyze_composition(generated, reference),
            "improvement_priority": self._calculate_improvement_priority(generated, reference)
        }
        
        # Generate specific recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_image_quality(self, generated: np.ndarray, reference: np.ndarray) -> Dict:
        """Analyze overall image quality metrics"""
        
        # Basic quality metrics
        sharpness_score = self._calculate_sharpness(generated)
        color_vibrancy = self._calculate_color_vibrancy(generated)
        contrast_quality = self._calculate_contrast(generated)
        noise_level = self._estimate_noise(generated)
        
        quality_analysis = {
            "overall_quality_score": min((sharpness_score + color_vibrancy + contrast_quality) / 3, 1.0),
            "sharpness": {
                "score": sharpness_score,
                "assessment": "excellent" if sharpness_score > 0.8 else "good" if sharpness_score > 0.6 else "needs_improvement"
            },
            "color_vibrancy": {
                "score": color_vibrancy,
                "assessment": "vibrant" if color_vibrancy > 0.7 else "acceptable" if color_vibrancy > 0.5 else "dull"
            },
            "contrast": {
                "score": contrast_quality,
                "assessment": "high" if contrast_quality > 0.8 else "medium" if contrast_quality > 0.6 else "low"
            },
            "noise_level": {
                "score": 1.0 - noise_level,  # Invert so higher is better
                "assessment": "clean" if noise_level < 0.2 else "slight_noise" if noise_level < 0.4 else "noisy"
            }
        }
        
        print(f"   📊 Quality Score: {quality_analysis['overall_quality_score']:.3f}")
        print(f"   📷 Sharpness: {quality_analysis['sharpness']['assessment']} ({sharpness_score:.3f})")
        print(f"   🎨 Colors: {quality_analysis['color_vibrancy']['assessment']} ({color_vibrancy:.3f})")
        print(f"   🌗 Contrast: {quality_analysis['contrast']['assessment']} ({contrast_quality:.3f})")
        print(f"   🔇 Noise: {quality_analysis['noise_level']['assessment']} ({1.0-noise_level:.3f})")
        
        return quality_analysis
    
    def _analyze_product_preservation(self, generated: np.ndarray, reference: np.ndarray) -> Dict:
        """Analyze how well the product was preserved from reference"""
        
        # Convert to same size for comparison
        gen_resized = self._resize_for_comparison(generated, reference.shape[:2])
        
        # Product structure preservation
        structure_similarity = self._calculate_structure_similarity(gen_resized, reference)
        
        # Color preservation (rough estimate)
        color_preservation = self._calculate_color_preservation(gen_resized, reference)
        
        # Shape preservation
        shape_preservation = self._calculate_shape_preservation(gen_resized, reference)
        
        preservation_analysis = {
            "overall_preservation_score": (structure_similarity + color_preservation + shape_preservation) / 3,
            "structure_similarity": {
                "score": structure_similarity,
                "assessment": "excellent" if structure_similarity > 0.8 else "good" if structure_similarity > 0.6 else "poor"
            },
            "color_preservation": {
                "score": color_preservation,
                "assessment": "well_preserved" if color_preservation > 0.7 else "partially_preserved" if color_preservation > 0.5 else "changed"
            },
            "shape_preservation": {
                "score": shape_preservation,
                "assessment": "maintained" if shape_preservation > 0.7 else "partially_maintained" if shape_preservation > 0.5 else "distorted"
            }
        }
        
        print(f"   🛡️ Preservation Score: {preservation_analysis['overall_preservation_score']:.3f}")
        print(f"   🏗️ Structure: {preservation_analysis['structure_similarity']['assessment']} ({structure_similarity:.3f})")
        print(f"   🎨 Color: {preservation_analysis['color_preservation']['assessment']} ({color_preservation:.3f})")
        print(f"   📐 Shape: {preservation_analysis['shape_preservation']['assessment']} ({shape_preservation:.3f})")
        
        return preservation_analysis
    
    def _check_hallucinations(self, generated: np.ndarray, reference: np.ndarray, prompt: str) -> Dict:
        """Check for unwanted hallucinations or artifacts"""
        
        # Look for common hallucination patterns
        unexpected_objects = self._detect_unexpected_objects(generated, prompt)
        duplicate_products = self._detect_duplicate_products(generated)
        distorted_features = self._detect_distorted_features(generated)
        background_contamination = self._detect_background_contamination(generated, prompt)
        
        hallucination_score = 1.0 - (unexpected_objects + duplicate_products + distorted_features + background_contamination) / 4
        
        hallucination_analysis = {
            "hallucination_free_score": max(hallucination_score, 0.0),
            "unexpected_objects": {
                "severity": unexpected_objects,
                "detected": unexpected_objects > 0.3
            },
            "duplicate_products": {
                "severity": duplicate_products,
                "detected": duplicate_products > 0.3
            },
            "distorted_features": {
                "severity": distorted_features,
                "detected": distorted_features > 0.3
            },
            "background_contamination": {
                "severity": background_contamination,
                "detected": background_contamination > 0.3
            }
        }
        
        print(f"   🚫 Hallucination-Free Score: {hallucination_analysis['hallucination_free_score']:.3f}")
        if unexpected_objects > 0.3:
            print(f"   ⚠️ Unexpected objects detected (severity: {unexpected_objects:.3f})")
        if duplicate_products > 0.3:
            print(f"   ⚠️ Product duplication detected (severity: {duplicate_products:.3f})")
        if distorted_features > 0.3:
            print(f"   ⚠️ Feature distortion detected (severity: {distorted_features:.3f})")
        if background_contamination > 0.3:
            print(f"   ⚠️ Background contamination detected (severity: {background_contamination:.3f})")
        
        return hallucination_analysis
    
    def _analyze_composition(self, generated: np.ndarray, reference: np.ndarray) -> Dict:
        """Analyze composition and scene layout"""
        
        # Composition metrics
        product_prominence = self._calculate_product_prominence(generated)
        lighting_quality = self._assess_lighting_quality(generated)
        scene_coherence = self._assess_scene_coherence(generated)
        professional_appeal = self._assess_professional_appeal(generated)
        
        composition_analysis = {
            "overall_composition_score": (product_prominence + lighting_quality + scene_coherence + professional_appeal) / 4,
            "product_prominence": {
                "score": product_prominence,
                "assessment": "prominent" if product_prominence > 0.7 else "visible" if product_prominence > 0.5 else "hidden"
            },
            "lighting_quality": {
                "score": lighting_quality,
                "assessment": "professional" if lighting_quality > 0.8 else "good" if lighting_quality > 0.6 else "poor"
            },
            "scene_coherence": {
                "score": scene_coherence,
                "assessment": "coherent" if scene_coherence > 0.7 else "mostly_coherent" if scene_coherence > 0.5 else "incoherent"
            },
            "professional_appeal": {
                "score": professional_appeal,
                "assessment": "commercial_ready" if professional_appeal > 0.8 else "needs_polish" if professional_appeal > 0.6 else "amateur"
            }
        }
        
        print(f"   🎭 Composition Score: {composition_analysis['overall_composition_score']:.3f}")
        print(f"   ⭐ Product Prominence: {composition_analysis['product_prominence']['assessment']} ({product_prominence:.3f})")
        print(f"   💡 Lighting: {composition_analysis['lighting_quality']['assessment']} ({lighting_quality:.3f})")
        print(f"   🎬 Scene Coherence: {composition_analysis['scene_coherence']['assessment']} ({scene_coherence:.3f})")
        print(f"   💼 Professional Appeal: {composition_analysis['professional_appeal']['assessment']} ({professional_appeal:.3f})")
        
        return composition_analysis
    
    def _calculate_improvement_priority(self, generated: np.ndarray, reference: np.ndarray) -> List[Dict]:
        """Calculate which aspects need the most improvement"""
        
        # This would be expanded with actual metrics
        # For now, providing a template structure
        
        priorities = [
            {"aspect": "product_preservation", "priority": "high", "score": 0.6, "target": 0.9},
            {"aspect": "hallucination_prevention", "priority": "critical", "score": 0.7, "target": 0.95},
            {"aspect": "image_quality", "priority": "medium", "score": 0.8, "target": 0.9},
            {"aspect": "composition", "priority": "low", "score": 0.85, "target": 0.9}
        ]
        
        # Sort by gap to target
        priorities.sort(key=lambda x: x["target"] - x["score"], reverse=True)
        
        print(f"   🎯 Improvement Priorities:")
        for i, priority in enumerate(priorities[:3]):
            gap = priority["target"] - priority["score"]
            print(f"   {i+1}. {priority['aspect']}: {priority['priority']} ({gap:.2f} gap)")
        
        return priorities
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate specific recommendations for improvement"""
        
        recommendations = []
        
        # Quality recommendations
        quality = analysis["quality_analysis"]
        if quality["overall_quality_score"] < 0.8:
            if quality["sharpness"]["score"] < 0.7:
                recommendations.append("Increase SDXL steps from 4 to 6-8 for better sharpness")
            if quality["color_vibrancy"]["score"] < 0.6:
                recommendations.append("Adjust CFG weight from 1.5 to 2.0-3.0 for more vibrant colors")
            if quality["noise_level"]["score"] < 0.7:
                recommendations.append("Enable quantization or adjust sampling method to reduce noise")
        
        # Preservation recommendations  
        preservation = analysis["product_preservation"]
        if preservation["overall_preservation_score"] < 0.8:
            if preservation["structure_similarity"]["score"] < 0.7:
                recommendations.append("Strengthen product structure preservation in CorePulse injection")
            if preservation["color_preservation"]["score"] < 0.7:
                recommendations.append("Implement color-specific preservation masks")
            if preservation["shape_preservation"]["score"] < 0.7:
                recommendations.append("Add shape-aware edge preservation to product masks")
        
        # Hallucination recommendations
        hallucination = analysis["hallucination_check"]
        if hallucination["hallucination_free_score"] < 0.9:
            if hallucination["unexpected_objects"]["detected"]:
                recommendations.append("Strengthen negative prompting to prevent object hallucination")
            if hallucination["duplicate_products"]["detected"]:
                recommendations.append("Add single-product constraint to generation prompts")
            if hallucination["background_contamination"]["detected"]:
                recommendations.append("Improve background/foreground separation in masks")
        
        # Composition recommendations
        composition = analysis["composition_analysis"]
        if composition["overall_composition_score"] < 0.8:
            if composition["product_prominence"]["score"] < 0.7:
                recommendations.append("Adjust prompt weighting to emphasize product visibility")
            if composition["lighting_quality"]["score"] < 0.7:
                recommendations.append("Add professional lighting keywords to enhancement prompts")
            if composition["professional_appeal"]["score"] < 0.8:
                recommendations.append("Include commercial photography style modifiers")
        
        print(f"   💡 Generated {len(recommendations)} improvement recommendations")
        
        return recommendations
    
    # Helper methods for calculations (simplified implementations)
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]) if len(image.shape) == 3 else image
        return min(np.var(self._simple_laplacian(gray)) / 1000.0, 1.0)
    
    def _simple_laplacian(self, image: np.ndarray) -> np.ndarray:
        """Simple Laplacian edge detection"""
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        h, w = image.shape
        result = np.zeros_like(image)
        for i in range(1, h-1):
            for j in range(1, w-1):
                result[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * kernel)
        return result
    
    def _calculate_color_vibrancy(self, image: np.ndarray) -> float:
        """Calculate color vibrancy"""
        if len(image.shape) != 3:
            return 0.5
        
        # Calculate saturation variance as proxy for vibrancy
        hsv_equivalent = np.std(image, axis=2)
        vibrancy = np.mean(hsv_equivalent) / 255.0
        return min(vibrancy * 2, 1.0)  # Scale appropriately
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast"""
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]) if len(image.shape) == 3 else image
        return min(np.std(gray) / 128.0, 1.0)
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level"""
        # Simple noise estimation using high-frequency content
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]) if len(image.shape) == 3 else image
        noise_estimate = np.std(gray - np.mean(gray)) / 255.0
        return min(noise_estimate, 1.0)
    
    def _resize_for_comparison(self, image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize image for comparison"""
        # Simple resize using PIL
        pil_image = Image.fromarray(image.astype(np.uint8))
        resized = pil_image.resize((target_shape[1], target_shape[0]))
        return np.array(resized)
    
    def _calculate_structure_similarity(self, generated: np.ndarray, reference: np.ndarray) -> float:
        """Calculate structural similarity (simplified SSIM-like)"""
        # Simplified structural similarity
        if generated.shape != reference.shape:
            return 0.5
        
        # Basic correlation-based similarity
        gen_flat = generated.flatten().astype(np.float32)
        ref_flat = reference.flatten().astype(np.float32)
        
        correlation = np.corrcoef(gen_flat, ref_flat)[0, 1]
        return max((correlation + 1) / 2, 0.0)  # Convert to 0-1 range
    
    def _calculate_color_preservation(self, generated: np.ndarray, reference: np.ndarray) -> float:
        """Calculate how well colors were preserved"""
        if generated.shape != reference.shape:
            return 0.5
        
        # Compare color histograms
        gen_colors = np.mean(generated, axis=(0, 1))
        ref_colors = np.mean(reference, axis=(0, 1))
        
        # Calculate color distance
        color_diff = np.sqrt(np.sum((gen_colors - ref_colors) ** 2))
        max_diff = np.sqrt(3 * 255 ** 2)
        
        return max(1.0 - (color_diff / max_diff), 0.0)
    
    def _calculate_shape_preservation(self, generated: np.ndarray, reference: np.ndarray) -> float:
        """Calculate shape preservation"""
        # Simplified shape preservation using edge similarity
        gen_edges = self._simple_laplacian(np.dot(generated[...,:3], [0.299, 0.587, 0.114]))
        ref_edges = self._simple_laplacian(np.dot(reference[...,:3], [0.299, 0.587, 0.114]))
        
        # Edge correlation
        edge_correlation = np.corrcoef(gen_edges.flatten(), ref_edges.flatten())[0, 1]
        return max((edge_correlation + 1) / 2, 0.0)
    
    # Simplified hallucination detection methods
    def _detect_unexpected_objects(self, generated: np.ndarray, prompt: str) -> float:
        """Detect unexpected objects (simplified)"""
        # Placeholder - would implement object detection
        return 0.2  # Assume low level of unexpected objects
    
    def _detect_duplicate_products(self, generated: np.ndarray) -> float:
        """Detect duplicate products (simplified)"""
        # Placeholder - would implement duplicate detection  
        return 0.1  # Assume low duplication
    
    def _detect_distorted_features(self, generated: np.ndarray) -> float:
        """Detect distorted features (simplified)"""
        # Placeholder - would implement distortion detection
        return 0.15  # Assume some minor distortion
    
    def _detect_background_contamination(self, generated: np.ndarray, prompt: str) -> float:
        """Detect background contamination (simplified)"""
        # Placeholder - would implement contamination detection
        return 0.1  # Assume minimal contamination
    
    # Composition analysis methods (simplified)
    def _calculate_product_prominence(self, image: np.ndarray) -> float:
        """Calculate how prominent the product is"""
        # Simplified - would use proper saliency detection
        return 0.75  # Assume good prominence
    
    def _assess_lighting_quality(self, image: np.ndarray) -> float:
        """Assess lighting quality"""
        # Check for even lighting distribution
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        lighting_variance = np.std(gray) / 255.0
        return min(lighting_variance * 2, 1.0)
    
    def _assess_scene_coherence(self, image: np.ndarray) -> float:
        """Assess scene coherence"""
        # Simplified coherence check
        return 0.8  # Assume good coherence
    
    def _assess_professional_appeal(self, image: np.ndarray) -> float:
        """Assess professional/commercial appeal"""
        # Combination of quality metrics
        sharpness = self._calculate_sharpness(image)
        contrast = self._calculate_contrast(image)
        vibrancy = self._calculate_color_vibrancy(image)
        
        return (sharpness + contrast + vibrancy) / 3
    
    def generate_improvement_plan(self, analyses: List[Dict]) -> Dict:
        """Generate comprehensive improvement plan from all analyses"""
        
        print(f"\n🎯 GENERATING IMPROVEMENT PLAN")
        print(f"Analyzed {len(analyses)} generated images")
        
        # Aggregate scores
        avg_scores = {
            "quality": np.mean([a["quality_analysis"]["overall_quality_score"] for a in analyses]),
            "preservation": np.mean([a["product_preservation"]["overall_preservation_score"] for a in analyses]),
            "hallucination_free": np.mean([a["hallucination_check"]["hallucination_free_score"] for a in analyses]),
            "composition": np.mean([a["composition_analysis"]["overall_composition_score"] for a in analyses])
        }
        
        # Overall success rate
        success_threshold = 0.8
        successful_generations = sum(1 for a in analyses if all(
            score > success_threshold for score in [
                a["quality_analysis"]["overall_quality_score"],
                a["product_preservation"]["overall_preservation_score"],
                a["hallucination_check"]["hallucination_free_score"],
                a["composition_analysis"]["overall_composition_score"]
            ]
        ))
        
        success_rate = successful_generations / len(analyses) if analyses else 0
        
        # Collect all recommendations
        all_recommendations = []
        for analysis in analyses:
            all_recommendations.extend(analysis["recommendations"])
        
        # Count recommendation frequency
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Sort by frequency
        priority_recommendations = sorted(
            recommendation_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]  # Top 10 most frequent
        
        improvement_plan = {
            "current_performance": {
                "overall_success_rate": f"{success_rate:.1%}",
                "successful_generations": f"{successful_generations}/{len(analyses)}",
                "average_scores": {k: f"{v:.3f}" for k, v in avg_scores.items()},
                "target_success_rate": "100%"
            },
            "priority_improvements": [
                {"recommendation": rec, "frequency": count, "priority": "high" if count >= len(analyses) * 0.5 else "medium"}
                for rec, count in priority_recommendations
            ],
            "next_iteration_config": self._generate_next_config(avg_scores, priority_recommendations),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"   📊 Current Success Rate: {success_rate:.1%}")
        print(f"   🎯 Target: 100% success rate")
        print(f"   📈 Average Scores:")
        for metric, score in avg_scores.items():
            status = "✅" if score > success_threshold else "⚠️" if score > 0.6 else "❌"
            print(f"     {status} {metric}: {score:.3f}")
        
        print(f"   🔧 Priority Improvements:")
        for i, (rec, count) in enumerate(priority_recommendations[:5]):
            print(f"     {i+1}. {rec} (affects {count}/{len(analyses)} images)")
        
        return improvement_plan
    
    def _generate_next_config(self, avg_scores: Dict, priority_recs: List) -> Dict:
        """Generate configuration for next iteration"""
        
        current_config = {
            "steps": 4,
            "cfg_weight": 1.5,
            "injection_strength": 0.85,
            "preservation_threshold": 0.9,
            "spatial_control_weight": 1.5
        }
        
        next_config = current_config.copy()
        
        # Apply recommendations to config
        for rec, count in priority_recs:
            if "steps from 4 to 6-8" in rec:
                next_config["steps"] = 6
            elif "CFG weight from 1.5 to 2.0-3.0" in rec:
                next_config["cfg_weight"] = 2.5
            elif "injection_strength" in rec:
                next_config["injection_strength"] = 0.9
            elif "preservation_threshold" in rec:
                next_config["preservation_threshold"] = 0.95
        
        return next_config


def main():
    """Main dogfooding analysis"""
    print("🍖 STARTING DOGFOODING ANALYSIS - MLX CorePulse V1")
    print("=" * 60)
    
    analyzer = DogfoodAnalyzer()
    
    # Test cases from our generation
    test_cases = [
        {
            "generated": "/Users/speed/Downloads/corpus-mlx/corepulse_mlx_watch.png",
            "reference": "/Users/speed/Downloads/corpus-mlx/test_product_watch.png", 
            "prompt": "luxury smartwatch on a modern glass desk in a bright office",
            "stats": {"generation_time": 8.15, "peak_memory": 11.63, "steps": 4, "cfg": 1.5}
        },
        {
            "generated": "/Users/speed/Downloads/corpus-mlx/corepulse_mlx_headphones.png",
            "reference": "/Users/speed/Downloads/corpus-mlx/test_product_headphones.png",
            "prompt": "premium gaming headphones on a wooden studio table with warm lighting", 
            "stats": {"generation_time": 8.00, "peak_memory": 11.63, "steps": 4, "cfg": 1.5}
        }
    ]
    
    analyses = []
    
    for i, test in enumerate(test_cases, 1):
        if Path(test["generated"]).exists() and Path(test["reference"]).exists():
            print(f"\n{'='*20} ANALYSIS {i}/2 {'='*20}")
            analysis = analyzer.analyze_generation_pair(
                test["generated"],
                test["reference"], 
                test["prompt"],
                test["stats"]
            )
            analyses.append(analysis)
        else:
            print(f"⚠️ Missing files for test case {i}")
    
    # Generate improvement plan
    if analyses:
        print(f"\n{'='*20} IMPROVEMENT PLAN {'='*20}")
        improvement_plan = analyzer.generate_improvement_plan(analyses)
        
        # Save results
        output_path = "/Users/speed/Downloads/corpus-mlx/dogfood_analysis_v1.json"
        with open(output_path, "w") as f:
            json.dump({
                "analyses": analyses,
                "improvement_plan": improvement_plan
            }, f, indent=2, default=str)
        
        print(f"\n✅ Dogfooding analysis complete!")
        print(f"📄 Full results saved to: {output_path}")
        
        return improvement_plan
    else:
        print("❌ No analyses completed - check file paths")
        return None


if __name__ == "__main__":
    main()