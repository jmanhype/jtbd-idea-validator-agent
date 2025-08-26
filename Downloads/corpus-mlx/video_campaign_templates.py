#!/usr/bin/env python3
"""
Video Campaign Templates
Business logic for different video ad campaign types
Templates for $100k/day Facebook ads scaling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
from pathlib import Path

class CampaignType(Enum):
    """Different campaign types for video ads"""
    PRODUCT_HERO = "product_hero"               # Product showcase
    LIFESTYLE_INTEGRATION = "lifestyle_integration"  # Product in lifestyle
    UNBOXING_REVEAL = "unboxing_reveal"        # Unboxing experience
    COMPARISON_DEMO = "comparison_demo"         # Before/after comparison
    SOCIAL_PROOF = "social_proof"              # User testimonials
    SEASONAL_PROMOTION = "seasonal_promotion"   # Holiday/seasonal themes
    PROBLEM_SOLUTION = "problem_solution"       # Problem → Product → Solution
    INFLUENCER_STYLE = "influencer_style"      # Influencer-style content


class Platform(Enum):
    """Target platforms for video campaigns"""
    FACEBOOK_FEED = "facebook_feed"
    INSTAGRAM_FEED = "instagram_feed" 
    INSTAGRAM_STORY = "instagram_story"
    INSTAGRAM_REELS = "instagram_reels"
    TIKTOK = "tiktok"
    YOUTUBE_SHORTS = "youtube_shorts"
    SNAPCHAT = "snapchat"
    TWITTER_VIDEO = "twitter_video"


class Demographic(Enum):
    """Target demographics"""
    GEN_Z = "gen_z"                    # 18-24
    MILLENNIALS = "millennials"         # 25-40
    GEN_X = "gen_x"                    # 41-56
    BOOMERS = "boomers"                # 57+
    PARENTS = "parents"                # Parents with kids
    PROFESSIONALS = "professionals"     # Working professionals
    STUDENTS = "students"              # Students


@dataclass
class VideoSpec:
    """Video specifications for different platforms"""
    width: int
    height: int
    duration_min: float  # seconds
    duration_max: float
    fps: int
    format: str
    aspect_ratio: str
    
    @property
    def is_vertical(self) -> bool:
        return self.height > self.width
    
    @property
    def is_square(self) -> bool:
        return self.height == self.width


@dataclass
class SceneTemplate:
    """Template for video scene composition"""
    name: str
    description: str
    background_type: str  # "studio", "lifestyle", "minimal", "gradient"
    product_placement: str  # "center", "left", "right", "hero", "lifestyle"
    lighting_style: str  # "dramatic", "natural", "soft", "studio"
    camera_movement: str  # "static", "zoom", "pan", "orbit"
    color_scheme: List[str]  # Primary colors for scene
    props: List[str]  # Additional props/elements
    text_overlays: List[Dict]  # Text positioning and style


@dataclass
class CampaignTemplate:
    """Complete campaign template"""
    type: CampaignType
    name: str
    description: str
    target_platforms: List[Platform]
    target_demographics: List[Demographic]
    scene_templates: List[SceneTemplate]
    video_specs: Dict[Platform, VideoSpec]
    conversion_elements: List[str]  # Elements that drive conversions
    estimated_ctr: Tuple[float, float]  # (min_ctr, max_ctr) range
    estimated_cpc: Tuple[float, float]  # (min_cpc, max_cpc) range
    scaling_potential: str  # "low", "medium", "high", "extreme"


class VideoCampaignTemplates:
    """
    Templates for high-converting video ad campaigns
    Optimized for $100k/day Facebook ads scaling
    """
    
    def __init__(self):
        self.platform_specs = self._initialize_platform_specs()
        self.conversion_patterns = self._initialize_conversion_patterns()
        self.templates = self._initialize_templates()
    
    def _initialize_platform_specs(self) -> Dict[Platform, VideoSpec]:
        """Initialize video specifications for each platform"""
        return {
            Platform.FACEBOOK_FEED: VideoSpec(
                width=1280, height=720, duration_min=6, duration_max=60,
                fps=30, format="mp4", aspect_ratio="16:9"
            ),
            Platform.INSTAGRAM_FEED: VideoSpec(
                width=1080, height=1080, duration_min=3, duration_max=60,
                fps=30, format="mp4", aspect_ratio="1:1"
            ),
            Platform.INSTAGRAM_STORY: VideoSpec(
                width=1080, height=1920, duration_min=3, duration_max=15,
                fps=30, format="mp4", aspect_ratio="9:16"
            ),
            Platform.INSTAGRAM_REELS: VideoSpec(
                width=1080, height=1920, duration_min=3, duration_max=90,
                fps=30, format="mp4", aspect_ratio="9:16"
            ),
            Platform.TIKTOK: VideoSpec(
                width=1080, height=1920, duration_min=9, duration_max=60,
                fps=30, format="mp4", aspect_ratio="9:16"
            ),
            Platform.YOUTUBE_SHORTS: VideoSpec(
                width=1080, height=1920, duration_min=15, duration_max=60,
                fps=30, format="mp4", aspect_ratio="9:16"
            ),
            Platform.SNAPCHAT: VideoSpec(
                width=1080, height=1920, duration_min=3, duration_max=180,
                fps=30, format="mp4", aspect_ratio="9:16"
            ),
            Platform.TWITTER_VIDEO: VideoSpec(
                width=1280, height=720, duration_min=6, duration_max=140,
                fps=30, format="mp4", aspect_ratio="16:9"
            )
        }
    
    def _initialize_conversion_patterns(self) -> Dict:
        """Patterns that drive high conversions"""
        return {
            "hook_patterns": [
                "You've been using [product] wrong",
                "This [product] changed my life",
                "Before you buy [product], watch this",
                "I tested [product] for 30 days",
                "Everyone's obsessed with this [product]",
                "[X] reasons you need [product]",
                "Stop buying [competitor], get [product]",
                "This [product] costs less than your [daily expense]"
            ],
            "pain_points": [
                "Tired of [problem]?",
                "Still struggling with [issue]?",
                "Fed up with [frustration]?",
                "Can't find a [solution] that works?",
                "Wasting money on [ineffective solution]?"
            ],
            "social_proof": [
                "[X] customers can't be wrong",
                "Rated #1 by [authority]",
                "Trending on [platform]",
                "[X]% recommend to friends",
                "Sold [X] units worldwide"
            ],
            "urgency_creators": [
                "Limited time offer",
                "Only [X] left in stock",
                "Price goes up tomorrow",
                "Black Friday exclusive",
                "First 100 customers only"
            ],
            "call_to_actions": [
                "Get yours now",
                "Shop the link",
                "Don't wait - order today",
                "Click to save [X]%",
                "Try risk-free for 30 days",
                "Free shipping ends soon"
            ]
        }
    
    def _initialize_templates(self) -> Dict[CampaignType, CampaignTemplate]:
        """Initialize all campaign templates"""
        templates = {}
        
        # Product Hero Template
        templates[CampaignType.PRODUCT_HERO] = CampaignTemplate(
            type=CampaignType.PRODUCT_HERO,
            name="Product Hero Showcase",
            description="Hero product shot with dramatic presentation",
            target_platforms=[Platform.INSTAGRAM_FEED, Platform.FACEBOOK_FEED, Platform.INSTAGRAM_REELS],
            target_demographics=[Demographic.MILLENNIALS, Demographic.PROFESSIONALS],
            scene_templates=[
                SceneTemplate(
                    name="Studio Hero",
                    description="Clean studio setup with dramatic lighting",
                    background_type="studio",
                    product_placement="hero",
                    lighting_style="dramatic",
                    camera_movement="zoom",
                    color_scheme=["#000000", "#FFFFFF", "#C9A961"],  # Black, white, gold
                    props=["pedestal", "soft_shadows"],
                    text_overlays=[
                        {"position": "top", "style": "bold", "content": "headline"},
                        {"position": "bottom", "style": "cta_button", "content": "call_to_action"}
                    ]
                ),
                SceneTemplate(
                    name="Minimal Luxury",
                    description="Minimal background with luxury feel",
                    background_type="minimal",
                    product_placement="center",
                    lighting_style="soft",
                    camera_movement="orbit",
                    color_scheme=["#F5F5F5", "#2C2C2C", "#DAA520"],  # Light gray, dark gray, gold
                    props=["marble_surface", "ambient_light"],
                    text_overlays=[
                        {"position": "center", "style": "elegant", "content": "brand_name"},
                        {"position": "bottom_right", "style": "minimal", "content": "price"}
                    ]
                )
            ],
            video_specs={},  # Will be populated
            conversion_elements=[
                "product_zoom", "feature_highlights", "premium_materials",
                "professional_lighting", "price_display", "satisfaction_guarantee"
            ],
            estimated_ctr=(2.5, 4.8),
            estimated_cpc=(0.45, 0.85),
            scaling_potential="high"
        )
        
        # Lifestyle Integration Template
        templates[CampaignType.LIFESTYLE_INTEGRATION] = CampaignTemplate(
            type=CampaignType.LIFESTYLE_INTEGRATION,
            name="Lifestyle Integration",
            description="Product naturally integrated into lifestyle scenes",
            target_platforms=[Platform.INSTAGRAM_STORY, Platform.TIKTOK, Platform.SNAPCHAT],
            target_demographics=[Demographic.GEN_Z, Demographic.MILLENNIALS],
            scene_templates=[
                SceneTemplate(
                    name="Morning Routine",
                    description="Product as part of morning routine",
                    background_type="lifestyle",
                    product_placement="lifestyle",
                    lighting_style="natural",
                    camera_movement="pan",
                    color_scheme=["#F8F8F8", "#E8E8E8", "#4A90E2"],  # Light tones, blue accent
                    props=["coffee", "sunlight", "plants", "modern_decor"],
                    text_overlays=[
                        {"position": "top_left", "style": "casual", "content": "time_stamp"},
                        {"position": "bottom", "style": "hashtag", "content": "lifestyle_tags"}
                    ]
                ),
                SceneTemplate(
                    name="Work From Home",
                    description="Product in professional home office",
                    background_type="lifestyle",
                    product_placement="left",
                    lighting_style="natural",
                    camera_movement="static",
                    color_scheme=["#FFFFFF", "#F0F0F0", "#5C7CFA"],  # Clean office colors
                    props=["laptop", "notebook", "plant", "clean_desk"],
                    text_overlays=[
                        {"position": "top_right", "style": "professional", "content": "productivity_stat"},
                        {"position": "bottom_center", "style": "cta", "content": "work_better"}
                    ]
                )
            ],
            video_specs={},
            conversion_elements=[
                "relatability", "natural_placement", "lifestyle_aspiration",
                "problem_solving", "daily_routine_fit", "authentic_usage"
            ],
            estimated_ctr=(3.2, 6.1),
            estimated_cpc=(0.32, 0.68),
            scaling_potential="extreme"
        )
        
        # Unboxing Reveal Template
        templates[CampaignType.UNBOXING_REVEAL] = CampaignTemplate(
            type=CampaignType.UNBOXING_REVEAL,
            name="Unboxing Experience",
            description="Engaging unboxing and first impression",
            target_platforms=[Platform.INSTAGRAM_REELS, Platform.TIKTOK, Platform.YOUTUBE_SHORTS],
            target_demographics=[Demographic.GEN_Z, Demographic.MILLENNIALS],
            scene_templates=[
                SceneTemplate(
                    name="ASMR Unboxing",
                    description="Satisfying unboxing with ASMR elements",
                    background_type="minimal",
                    product_placement="center",
                    lighting_style="soft",
                    camera_movement="zoom",
                    color_scheme=["#FFFFFF", "#F5F5F5", "#FF6B6B"],  # Clean with accent
                    props=["clean_surface", "packaging", "soft_materials"],
                    text_overlays=[
                        {"position": "top", "style": "excitement", "content": "unboxing_text"},
                        {"position": "bottom", "style": "price_reveal", "content": "value_prop"}
                    ]
                )
            ],
            video_specs={},
            conversion_elements=[
                "anticipation_build", "packaging_quality", "first_impression",
                "value_reveal", "excitement_factor", "shareability"
            ],
            estimated_ctr=(4.1, 7.8),
            estimated_cpc=(0.28, 0.55),
            scaling_potential="extreme"
        )
        
        # Problem-Solution Template
        templates[CampaignType.PROBLEM_SOLUTION] = CampaignTemplate(
            type=CampaignType.PROBLEM_SOLUTION,
            name="Problem-Solution Story",
            description="Clear problem identification and solution presentation",
            target_platforms=[Platform.FACEBOOK_FEED, Platform.INSTAGRAM_FEED, Platform.YOUTUBE_SHORTS],
            target_demographics=[Demographic.MILLENNIALS, Demographic.GEN_X, Demographic.PARENTS],
            scene_templates=[
                SceneTemplate(
                    name="Before After",
                    description="Visual before/after transformation",
                    background_type="studio",
                    product_placement="hero",
                    lighting_style="dramatic",
                    camera_movement="pan",
                    color_scheme=["#FF4757", "#2ED573", "#FFFFFF"],  # Red problem, green solution
                    props=["comparison_setup", "dramatic_lighting"],
                    text_overlays=[
                        {"position": "left", "style": "problem", "content": "before_state"},
                        {"position": "right", "style": "solution", "content": "after_state"},
                        {"position": "bottom", "style": "cta", "content": "get_solution"}
                    ]
                )
            ],
            video_specs={},
            conversion_elements=[
                "problem_agitation", "clear_solution", "immediate_benefit",
                "visual_proof", "emotional_connection", "urgency_creation"
            ],
            estimated_ctr=(3.8, 6.9),
            estimated_cpc=(0.38, 0.72),
            scaling_potential="high"
        )
        
        # Social Proof Template
        templates[CampaignType.SOCIAL_PROOF] = CampaignTemplate(
            type=CampaignType.SOCIAL_PROOF,
            name="Social Proof Showcase",
            description="Customer testimonials and reviews",
            target_platforms=[Platform.FACEBOOK_FEED, Platform.INSTAGRAM_FEED, Platform.TWITTER_VIDEO],
            target_demographics=[Demographic.MILLENNIALS, Demographic.GEN_X, Demographic.PROFESSIONALS],
            scene_templates=[
                SceneTemplate(
                    name="Review Compilation",
                    description="Multiple customer reviews and ratings",
                    background_type="gradient",
                    product_placement="left",
                    lighting_style="natural",
                    camera_movement="static",
                    color_scheme=["#4CAF50", "#FFFFFF", "#FFC107"],  # Green, white, gold (ratings)
                    props=["review_cards", "star_ratings", "customer_photos"],
                    text_overlays=[
                        {"position": "right", "style": "testimonial", "content": "customer_quote"},
                        {"position": "bottom", "style": "stats", "content": "rating_stats"},
                        {"position": "bottom_right", "style": "cta", "content": "join_customers"}
                    ]
                )
            ],
            video_specs={},
            conversion_elements=[
                "customer_testimonials", "rating_displays", "usage_statistics",
                "trust_indicators", "peer_validation", "risk_reduction"
            ],
            estimated_ctr=(2.9, 5.4),
            estimated_cpc=(0.42, 0.79),
            scaling_potential="medium"
        )
        
        # Populate video specs for all templates
        for template in templates.values():
            template.video_specs = {
                platform: self.platform_specs[platform] 
                for platform in template.target_platforms
            }
        
        return templates
    
    def get_template(self, campaign_type: CampaignType) -> CampaignTemplate:
        """Get specific campaign template"""
        return self.templates.get(campaign_type)
    
    def get_templates_for_platform(self, platform: Platform) -> List[CampaignTemplate]:
        """Get all templates suitable for specific platform"""
        return [
            template for template in self.templates.values()
            if platform in template.target_platforms
        ]
    
    def get_templates_for_demographic(self, demographic: Demographic) -> List[CampaignTemplate]:
        """Get all templates suitable for specific demographic"""
        return [
            template for template in self.templates.values()
            if demographic in template.target_demographics
        ]
    
    def generate_campaign_variations(
        self,
        base_template: CampaignTemplate,
        variation_count: int = 5
    ) -> List[Dict]:
        """
        Generate multiple variations of a campaign template for A/B testing
        """
        print(f"🎬 Generating {variation_count} variations for {base_template.name}")
        
        variations = []
        
        for i in range(variation_count):
            # Create variation by modifying elements
            variation = {
                "variation_id": f"{base_template.type.value}_v{i+1}",
                "base_template": base_template.type.value,
                "modifications": self._generate_variation_modifications(base_template, i),
                "expected_performance": self._calculate_variation_performance(base_template, i),
                "targeting_adjustments": self._generate_targeting_adjustments(i),
                "creative_elements": self._generate_creative_variations(base_template, i)
            }
            variations.append(variation)
            
            print(f"   ✅ Variation {i+1}: {variation['expected_performance']['ctr_multiplier']:.2f}x CTR expected")
        
        return variations
    
    def _generate_variation_modifications(
        self,
        template: CampaignTemplate,
        variation_index: int
    ) -> Dict:
        """Generate specific modifications for variation"""
        modifications = {
            "scene_adjustments": {},
            "text_changes": {},
            "color_scheme_shifts": {},
            "timing_adjustments": {},
            "platform_optimizations": {}
        }
        
        # Rotate through different modification patterns
        mod_pattern = variation_index % 4
        
        if mod_pattern == 0:
            # Focus on color scheme changes
            modifications["color_scheme_shifts"] = {
                "primary_color": self._get_alternative_color_scheme(variation_index),
                "mood": "warmer" if variation_index % 2 == 0 else "cooler"
            }
        elif mod_pattern == 1:
            # Focus on text and messaging
            modifications["text_changes"] = {
                "hook_style": self._select_hook_pattern(variation_index),
                "cta_variation": self._select_cta_variation(variation_index),
                "urgency_level": "high" if variation_index > 2 else "medium"
            }
        elif mod_pattern == 2:
            # Focus on scene composition
            modifications["scene_adjustments"] = {
                "product_position": ["left", "center", "right"][variation_index % 3],
                "camera_angle": ["low", "eye_level", "high"][variation_index % 3],
                "background_complexity": "minimal" if variation_index % 2 == 0 else "detailed"
            }
        else:
            # Focus on timing and pacing
            modifications["timing_adjustments"] = {
                "intro_duration": 1.0 + (variation_index * 0.5),
                "product_reveal_timing": 2.0 + (variation_index * 0.3),
                "cta_duration": 3.0 + (variation_index * 0.2)
            }
        
        return modifications
    
    def _calculate_variation_performance(
        self,
        base_template: CampaignTemplate,
        variation_index: int
    ) -> Dict:
        """Calculate expected performance for variation"""
        base_ctr_min, base_ctr_max = base_template.estimated_ctr
        base_cpc_min, base_cpc_max = base_template.estimated_cpc
        
        # Apply variation multipliers
        variation_multipliers = [1.0, 0.85, 1.15, 0.92, 1.08]  # Different performance expectations
        multiplier = variation_multipliers[variation_index % len(variation_multipliers)]
        
        return {
            "ctr_multiplier": multiplier,
            "estimated_ctr_range": (base_ctr_min * multiplier, base_ctr_max * multiplier),
            "estimated_cpc_range": (base_cpc_min / multiplier, base_cpc_max / multiplier),
            "confidence_level": max(0.6, 1.0 - abs(1.0 - multiplier) * 2)
        }
    
    def _generate_targeting_adjustments(self, variation_index: int) -> Dict:
        """Generate targeting adjustments for variation"""
        adjustments = {
            "age_range_shift": 0,
            "interest_expansion": False,
            "lookalike_percentage": 1,  # 1%, 2%, 5%, 10%
            "device_targeting": "all"
        }
        
        if variation_index == 1:
            adjustments["age_range_shift"] = -3  # Target slightly younger
            adjustments["device_targeting"] = "mobile_heavy"
        elif variation_index == 2:
            adjustments["age_range_shift"] = +5  # Target slightly older
            adjustments["interest_expansion"] = True
        elif variation_index == 3:
            adjustments["lookalike_percentage"] = 2
        elif variation_index == 4:
            adjustments["lookalike_percentage"] = 5
            adjustments["interest_expansion"] = True
        
        return adjustments
    
    def _generate_creative_variations(
        self,
        template: CampaignTemplate,
        variation_index: int
    ) -> Dict:
        """Generate creative element variations"""
        base_elements = template.conversion_elements
        
        # Add variation-specific elements
        variation_elements = {
            0: ["premium_badge", "limited_edition"],
            1: ["social_media_buzz", "trending_indicator"],
            2: ["money_back_guarantee", "free_shipping"],
            3: ["customer_count", "satisfaction_rate"],
            4: ["urgency_timer", "stock_counter"]
        }
        
        additional_elements = variation_elements.get(variation_index, [])
        
        return {
            "base_elements": base_elements,
            "additional_elements": additional_elements,
            "emphasis_elements": base_elements[:2],  # Emphasize first 2 elements
            "creative_style": ["clean", "bold", "playful", "elegant", "dramatic"][variation_index % 5]
        }
    
    def _get_alternative_color_scheme(self, variation_index: int) -> List[str]:
        """Get alternative color schemes"""
        schemes = [
            ["#FF6B6B", "#4ECDC4", "#FFFFFF"],  # Red-teal
            ["#6C5CE7", "#FDCB6E", "#FFFFFF"],  # Purple-yellow
            ["#00B894", "#FFFFFF", "#2D3436"],  # Green-black
            ["#E17055", "#74B9FF", "#FFFFFF"],  # Orange-blue
            ["#FD79A8", "#FDCB6E", "#FFFFFF"]   # Pink-yellow
        ]
        return schemes[variation_index % len(schemes)]
    
    def _select_hook_pattern(self, variation_index: int) -> str:
        """Select hook pattern for variation"""
        hooks = self.conversion_patterns["hook_patterns"]
        return hooks[variation_index % len(hooks)]
    
    def _select_cta_variation(self, variation_index: int) -> str:
        """Select CTA variation"""
        ctas = self.conversion_patterns["call_to_actions"]
        return ctas[variation_index % len(ctas)]
    
    def create_campaign_deployment_package(
        self,
        template: CampaignTemplate,
        variations: List[Dict],
        target_budget: float = 1000.0
    ) -> Dict:
        """
        Create complete deployment package for RunPod processing
        """
        print(f"📦 Creating deployment package for {template.name}")
        print(f"   Budget: ${target_budget:.2f}")
        print(f"   Variations: {len(variations)}")
        
        package = {
            "campaign_info": {
                "template_name": template.name,
                "campaign_type": template.type.value,
                "target_budget": target_budget,
                "expected_performance": {
                    "ctr_range": template.estimated_ctr,
                    "cpc_range": template.estimated_cpc,
                    "scaling_potential": template.scaling_potential
                }
            },
            "technical_specs": {
                "platform_requirements": {
                    platform.value: asdict(spec) 
                    for platform, spec in template.video_specs.items()
                },
                "scene_templates": [
                    asdict(scene) for scene in template.scene_templates
                ],
                "conversion_elements": template.conversion_elements
            },
            "variations": variations,
            "production_requirements": {
                "total_videos_needed": len(variations) * len(template.target_platforms),
                "estimated_production_time": len(variations) * 2.5,  # hours
                "resource_requirements": {
                    "gpu_memory": "24GB+",
                    "storage": f"{len(variations) * 2}GB",
                    "processing_time": f"{len(variations) * 10}min"
                }
            },
            "deployment_schedule": {
                "phase_1": {
                    "variations": variations[:2],
                    "budget_split": 0.3,
                    "duration": "24 hours",
                    "success_criteria": "CTR > base_template minimum"
                },
                "phase_2": {
                    "variations": variations[2:],
                    "budget_split": 0.7,
                    "duration": "72 hours", 
                    "success_criteria": "CPA < target threshold"
                }
            },
            "metadata": {
                "generated_on": "mac",
                "package_version": "1.0",
                "hash": hashlib.md5(
                    f"{template.name}_{len(variations)}_{target_budget}".encode()
                ).hexdigest()[:12]
            }
        }
        
        print(f"   ✅ Package created with hash: {package['metadata']['hash']}")
        return package


def main():
    """Test video campaign templates"""
    print("\n🎬 Testing Video Campaign Templates (Mac)")
    print("="*55)
    
    # Initialize templates
    templates = VideoCampaignTemplates()
    
    # Test 1: List all templates
    print("\n1. Available Campaign Templates:")
    for campaign_type, template in templates.templates.items():
        print(f"   {template.name}:")
        print(f"     Type: {campaign_type.value}")
        print(f"     Platforms: {len(template.target_platforms)}")
        print(f"     Demographics: {len(template.target_demographics)}")
        print(f"     CTR Range: {template.estimated_ctr[0]:.1f}%-{template.estimated_ctr[1]:.1f}%")
        print(f"     Scaling: {template.scaling_potential}")
    
    # Test 2: Get template and generate variations
    print(f"\n2. Testing Product Hero Template...")
    hero_template = templates.get_template(CampaignType.PRODUCT_HERO)
    
    if hero_template:
        print(f"   Template: {hero_template.name}")
        print(f"   Scene templates: {len(hero_template.scene_templates)}")
        
        # Generate variations
        variations = templates.generate_campaign_variations(hero_template, variation_count=3)
        
        for i, variation in enumerate(variations):
            print(f"   Variation {i+1}: {variation['variation_id']}")
            print(f"     Expected CTR multiplier: {variation['expected_performance']['ctr_multiplier']:.2f}x")
    
    # Test 3: Platform-specific templates
    print(f"\n3. Templates for Instagram Reels:")
    reels_templates = templates.get_templates_for_platform(Platform.INSTAGRAM_REELS)
    for template in reels_templates:
        print(f"   - {template.name} (CTR: {template.estimated_ctr[0]:.1f}%-{template.estimated_ctr[1]:.1f}%)")
    
    # Test 4: Demographic-specific templates
    print(f"\n4. Templates for Gen Z:")
    genz_templates = templates.get_templates_for_demographic(Demographic.GEN_Z)
    for template in genz_templates:
        print(f"   - {template.name} (Scaling: {template.scaling_potential})")
    
    # Test 5: Create deployment package
    print(f"\n5. Creating deployment package...")
    if hero_template:
        variations = templates.generate_campaign_variations(hero_template, variation_count=5)
        package = templates.create_campaign_deployment_package(
            template=hero_template,
            variations=variations,
            target_budget=5000.0
        )
        
        # Save package
        with open("campaign_deployment_package.json", 'w') as f:
            json.dump(package, f, indent=2)
        
        print(f"   ✅ Package saved with {package['production_requirements']['total_videos_needed']} videos needed")
        print(f"   ✅ Estimated production time: {package['production_requirements']['estimated_production_time']} hours")
    
    print("\n✅ Video Campaign Templates Test Complete!")
    print("🎬 Ready for $100k/day scaling deployment")


if __name__ == "__main__":
    main()