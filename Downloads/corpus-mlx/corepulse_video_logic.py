#!/usr/bin/env python3
"""
CorePulse Video Control Logic
Algorithms for frame-by-frame control WITHOUT video generation
This runs on Mac - RunPod handles the actual inference
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import hashlib

class FrameInjectionLevel(Enum):
    """Video-specific injection levels"""
    TEMPORAL_EARLY = "temporal_early"    # Frame sequence structure
    SPATIAL_MID = "spatial_mid"          # Spatial relationships  
    STYLE_LATE = "style_late"            # Lighting/color consistency
    TEMPORAL_CONSISTENCY = "temporal_consistency"  # Frame-to-frame coherence


@dataclass
class VideoFrameInjection:
    """Frame-specific injection control"""
    frame_index: int
    prompt: str
    level: FrameInjectionLevel
    strength: float
    region_mask: Optional[np.ndarray] = None
    temporal_weight: float = 1.0  # How much this affects neighboring frames


@dataclass
class ProductLockRegion:
    """Region to preserve exactly across frames"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    frame_start: int
    frame_end: int
    lock_strength: float = 1.0  # 1.0 = absolute preservation
    feather_radius: int = 0     # Edge softening


@dataclass
class TemporalConsistencyRule:
    """Rules for maintaining consistency across frames"""
    attribute: str  # 'color', 'shape', 'position', 'scale'
    tolerance: float  # Allowed variation (0.0 = no change, 1.0 = any change)
    smoothing: float  # Temporal smoothing factor (0.0 = no smooth, 1.0 = max smooth)
    priority: int = 1  # Higher number = higher priority


class CorePulseVideoLogic:
    """
    Video control algorithms for CorePulse
    NO ACTUAL VIDEO GENERATION - just the control logic
    """
    
    def __init__(self):
        self.frame_injections = []
        self.product_locks = []
        self.consistency_rules = []
        self.frame_count = 0
        
        # Pre-calculated injection templates
        self.templates = self._load_injection_templates()
    
    def _load_injection_templates(self) -> Dict:
        """Pre-defined injection patterns for common scenarios"""
        return {
            "product_hero_shot": {
                "early": "product positioning, stable placement",
                "mid": "clean background, proper lighting setup", 
                "late": "professional product photography lighting"
            },
            "lifestyle_scene": {
                "early": "natural environment, realistic placement",
                "mid": "lifestyle context, ambient scene",
                "late": "natural lighting, authentic atmosphere"
            },
            "luxury_showcase": {
                "early": "premium surface, elegant positioning",
                "mid": "luxury environment, sophisticated context",
                "late": "dramatic lighting, premium atmosphere"
            },
            "social_media_ad": {
                "early": "eye-catching composition, social format",
                "mid": "engaging background, platform-optimized",
                "late": "vibrant colors, social media aesthetic"
            }
        }
    
    def calculate_frame_injections(
        self,
        video_length: int,
        template_name: str,
        product_regions: List[ProductLockRegion],
        custom_prompts: Optional[Dict] = None
    ) -> List[VideoFrameInjection]:
        """
        Calculate injection schedule for entire video
        This is the CORE algorithm that runs on Mac
        """
        injections = []
        template = self.templates.get(template_name, self.templates["product_hero_shot"])
        
        print(f"\n🧠 Calculating injections for {video_length} frames...")
        print(f"   Template: {template_name}")
        print(f"   Product regions: {len(product_regions)}")
        
        for frame_idx in range(video_length):
            # Calculate frame position (0.0 = start, 1.0 = end)
            position = frame_idx / max(1, video_length - 1)
            
            # Early injection (structure/positioning)
            early_strength = self._calculate_temporal_strength(
                position, phase="early"
            )
            injections.append(VideoFrameInjection(
                frame_index=frame_idx,
                prompt=template["early"],
                level=FrameInjectionLevel.TEMPORAL_EARLY,
                strength=early_strength,
                temporal_weight=0.8  # Affects 80% of neighboring frames
            ))
            
            # Mid injection (context/environment)  
            mid_strength = self._calculate_temporal_strength(
                position, phase="mid"
            )
            injections.append(VideoFrameInjection(
                frame_index=frame_idx,
                prompt=template["mid"],
                level=FrameInjectionLevel.SPATIAL_MID,
                strength=mid_strength,
                temporal_weight=0.6
            ))
            
            # Late injection (style/lighting)
            late_strength = self._calculate_temporal_strength(
                position, phase="late"
            )
            injections.append(VideoFrameInjection(
                frame_index=frame_idx,
                prompt=template["late"],
                level=FrameInjectionLevel.STYLE_LATE,
                strength=late_strength,
                temporal_weight=0.4
            ))
            
            # Product-specific injections
            for region in product_regions:
                if region.frame_start <= frame_idx <= region.frame_end:
                    injections.append(VideoFrameInjection(
                        frame_index=frame_idx,
                        prompt="preserve exact product pixels, zero hallucination",
                        level=FrameInjectionLevel.TEMPORAL_CONSISTENCY,
                        strength=region.lock_strength,
                        region_mask=self._create_region_mask(region.bbox),
                        temporal_weight=1.0  # Maximum temporal influence
                    ))
        
        print(f"   ✅ Generated {len(injections)} injections")
        return injections
    
    def _calculate_temporal_strength(
        self,
        position: float,
        phase: str
    ) -> float:
        """
        Calculate injection strength based on temporal position
        Different phases have different importance curves
        """
        if phase == "early":
            # Strong at beginning and end (structure important)
            strength = 0.9 - 0.3 * np.sin(position * np.pi)
        elif phase == "mid":
            # Consistent throughout (context always important)
            strength = 0.8 + 0.1 * np.sin(position * np.pi * 2)
        elif phase == "late":
            # Builds up over time (style refinement)
            strength = 0.5 + 0.4 * position
        else:
            strength = 0.7
        
        return max(0.1, min(1.0, strength))
    
    def _create_region_mask(
        self,
        bbox: Tuple[int, int, int, int],
        frame_size: Tuple[int, int] = (1024, 1024)
    ) -> np.ndarray:
        """Create binary mask for region locking"""
        mask = np.zeros(frame_size, dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
        return mask
    
    def calculate_temporal_consistency(
        self,
        product_regions: List[ProductLockRegion],
        video_length: int
    ) -> List[TemporalConsistencyRule]:
        """
        Calculate consistency rules for temporal coherence
        Ensures products don't change between frames
        """
        rules = []
        
        for region in product_regions:
            # Color consistency rule
            rules.append(TemporalConsistencyRule(
                attribute="color",
                tolerance=0.02,  # 2% color variation max
                smoothing=0.8,   # High smoothing
                priority=10      # Very high priority
            ))
            
            # Shape consistency rule
            rules.append(TemporalConsistencyRule(
                attribute="shape", 
                tolerance=0.01,  # 1% shape variation max
                smoothing=0.9,   # Maximum smoothing
                priority=10      # Very high priority
            ))
            
            # Position smoothing rule
            rules.append(TemporalConsistencyRule(
                attribute="position",
                tolerance=0.05,  # 5% position variation
                smoothing=0.7,   # Moderate smoothing
                priority=8       # High priority
            ))
            
            # Scale consistency rule
            rules.append(TemporalConsistencyRule(
                attribute="scale",
                tolerance=0.03,  # 3% scale variation max
                smoothing=0.8,   # High smoothing
                priority=9       # Very high priority
            ))
        
        print(f"   ✅ Generated {len(rules)} consistency rules")
        return rules
    
    def generate_injection_schedule(
        self,
        video_config: Dict,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate complete injection schedule for video
        This is what gets sent to RunPod for execution
        """
        video_length = video_config["frame_count"]
        template_name = video_config["template"]
        
        # Parse product regions
        product_regions = []
        for region_data in video_config.get("product_regions", []):
            product_regions.append(ProductLockRegion(
                bbox=tuple(region_data["bbox"]),
                frame_start=region_data["frame_start"],
                frame_end=region_data["frame_end"],
                lock_strength=region_data.get("lock_strength", 1.0),
                feather_radius=region_data.get("feather_radius", 0)
            ))
        
        # Calculate all injections
        injections = self.calculate_frame_injections(
            video_length=video_length,
            template_name=template_name,
            product_regions=product_regions,
            custom_prompts=video_config.get("custom_prompts")
        )
        
        # Calculate consistency rules
        consistency_rules = self.calculate_temporal_consistency(
            product_regions=product_regions,
            video_length=video_length
        )
        
        # Create deployment package
        schedule = {
            "version": "1.0",
            "video_config": video_config,
            "frame_injections": [
                {
                    "frame_index": inj.frame_index,
                    "prompt": inj.prompt,
                    "level": inj.level.value,
                    "strength": inj.strength,
                    "temporal_weight": inj.temporal_weight,
                    "region_mask": inj.region_mask.tolist() if inj.region_mask is not None else None
                }
                for inj in injections
            ],
            "consistency_rules": [
                {
                    "attribute": rule.attribute,
                    "tolerance": rule.tolerance,
                    "smoothing": rule.smoothing,
                    "priority": rule.priority
                }
                for rule in consistency_rules
            ],
            "product_locks": [
                {
                    "bbox": region.bbox,
                    "frame_start": region.frame_start,
                    "frame_end": region.frame_end,
                    "lock_strength": region.lock_strength,
                    "feather_radius": region.feather_radius
                }
                for region in product_regions
            ],
            "metadata": {
                "generated_on": "mac",
                "total_injections": len(injections),
                "total_rules": len(consistency_rules),
                "hash": self._generate_schedule_hash(injections, consistency_rules)
            }
        }
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(schedule, f, indent=2)
            print(f"   ✅ Schedule saved to: {output_path}")
        
        return schedule
    
    def _generate_schedule_hash(
        self,
        injections: List[VideoFrameInjection],
        rules: List[TemporalConsistencyRule]
    ) -> str:
        """Generate hash for schedule validation"""
        content = f"{len(injections)}_{len(rules)}_{injections[0].prompt if injections else ''}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def validate_schedule(self, schedule: Dict) -> Tuple[bool, List[str]]:
        """
        Validate injection schedule for deployment
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        required_fields = ["frame_injections", "consistency_rules", "product_locks"]
        for field in required_fields:
            if field not in schedule:
                issues.append(f"Missing required field: {field}")
        
        # Check injection validity
        if "frame_injections" in schedule:
            injections = schedule["frame_injections"]
            if not injections:
                issues.append("No frame injections defined")
            
            # Check injection strengths
            for i, inj in enumerate(injections):
                if not 0.0 <= inj.get("strength", 0) <= 1.0:
                    issues.append(f"Injection {i}: Invalid strength {inj.get('strength')}")
        
        # Check consistency rules
        if "consistency_rules" in schedule:
            rules = schedule["consistency_rules"]
            for i, rule in enumerate(rules):
                if not 0.0 <= rule.get("tolerance", 1) <= 1.0:
                    issues.append(f"Rule {i}: Invalid tolerance {rule.get('tolerance')}")
        
        # Check product locks
        if "product_locks" in schedule:
            locks = schedule["product_locks"]
            for i, lock in enumerate(locks):
                if lock.get("lock_strength", 0) < 0.5:
                    issues.append(f"Lock {i}: Low lock strength {lock.get('lock_strength')} may allow hallucination")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def create_test_schedule(
        self,
        frames: int = 30,
        product_bbox: Tuple[int, int, int, int] = (400, 400, 600, 600)
    ) -> Dict:
        """Create test schedule for validation"""
        test_config = {
            "frame_count": frames,
            "template": "product_hero_shot",
            "product_regions": [
                {
                    "bbox": list(product_bbox),
                    "frame_start": 0,
                    "frame_end": frames - 1,
                    "lock_strength": 1.0,
                    "feather_radius": 0
                }
            ]
        }
        
        return self.generate_injection_schedule(test_config)


def main():
    """Test the video logic algorithms"""
    print("\n🧠 Testing CorePulse Video Logic (Mac)")
    print("="*50)
    
    # Initialize logic controller
    logic = CorePulseVideoLogic()
    
    # Test 1: Create injection schedule
    print("\n1. Testing injection schedule generation...")
    test_schedule = logic.create_test_schedule(frames=10)
    print(f"   Generated schedule with {len(test_schedule['frame_injections'])} injections")
    
    # Test 2: Validate schedule
    print("\n2. Testing schedule validation...")
    is_valid, issues = logic.validate_schedule(test_schedule)
    print(f"   Schedule valid: {is_valid}")
    if issues:
        for issue in issues:
            print(f"   ⚠️ {issue}")
    
    # Test 3: Different templates
    print("\n3. Testing different templates...")
    templates = ["product_hero_shot", "lifestyle_scene", "luxury_showcase", "social_media_ad"]
    
    for template in templates:
        config = {
            "frame_count": 5,
            "template": template,
            "product_regions": []
        }
        schedule = logic.generate_injection_schedule(config)
        print(f"   {template}: {len(schedule['frame_injections'])} injections")
    
    # Test 4: Save test schedule
    print("\n4. Saving test schedule...")
    logic.generate_injection_schedule(
        test_schedule["video_config"],
        output_path="test_video_schedule.json"
    )
    
    print("\n✅ CorePulse Video Logic Test Complete!")
    print("📦 Ready for RunPod deployment")


if __name__ == "__main__":
    main()