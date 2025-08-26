#!/usr/bin/env python3
"""
AGC (Artificial General Commerce) Ads Pipeline
Production-ready system for generating hallucination-free product ads at scale
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import mlx.core as mx
from stable_diffusion import StableDiffusionXL
from corepulse_mlx import CorePulseMLX, PromptInjection, InjectionLevel, TokenMask, SpatialInjection
from hallucination_free_placement import HallucinationFreeProductPlacement
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import uuid


@dataclass
class AdCampaign:
    """Ad campaign configuration"""
    campaign_id: str
    brand_name: str
    product_category: str
    target_audience: str
    brand_colors: List[str]
    brand_voice: str  # luxury, casual, professional, playful
    compliance_requirements: List[str] = None


@dataclass
class AdFormat:
    """Standard ad format specifications"""
    name: str
    width: int
    height: int
    platform: str  # instagram, facebook, google, tiktok, youtube
    safe_zones: Dict[str, Tuple[int, int, int, int]] = None  # Areas to avoid for text/logos
    cta_position: str = "bottom_right"  # bottom_right, bottom_center, top_right
    max_text_length: int = 50


@dataclass
class ProductAd:
    """Individual ad configuration"""
    ad_id: str
    campaign: AdCampaign
    product_image: str
    headline: str
    subheadline: Optional[str]
    cta_text: str  # Call-to-action
    scene_theme: str
    format: AdFormat
    variants: List[str] = None  # A/B test variants
    performance_metrics: Dict = None


class AGCAdsEngine:
    """
    Production-grade AGC Ads Engine
    Generates hallucination-free product ads with:
    - Multi-format support
    - A/B testing variants
    - Brand consistency
    - Performance tracking
    - Batch processing
    """
    
    # Standard ad formats
    AD_FORMATS = {
        "instagram_feed": AdFormat("Instagram Feed", 1080, 1080, "instagram"),
        "instagram_story": AdFormat("Instagram Story", 1080, 1920, "instagram"),
        "facebook_feed": AdFormat("Facebook Feed", 1200, 630, "facebook"),
        "google_display": AdFormat("Google Display", 300, 250, "google"),
        "youtube_thumbnail": AdFormat("YouTube Thumbnail", 1280, 720, "youtube"),
        "tiktok_video": AdFormat("TikTok Video", 1080, 1920, "tiktok"),
        "twitter_card": AdFormat("Twitter Card", 1200, 675, "twitter"),
        "linkedin_sponsored": AdFormat("LinkedIn Sponsored", 1200, 627, "linkedin"),
    }
    
    # Scene themes optimized for conversions
    CONVERSION_OPTIMIZED_SCENES = {
        "luxury": {
            "surfaces": ["marble", "glass", "polished wood"],
            "lighting": ["dramatic", "studio"],
            "moods": ["luxury", "minimal"],
            "props": ["gold accents", "velvet", "crystal"]
        },
        "lifestyle": {
            "surfaces": ["table", "desk", "shelf"],
            "lighting": ["natural", "soft"],
            "moods": ["modern", "cozy"],
            "props": ["plants", "coffee", "books"]
        },
        "tech": {
            "surfaces": ["desk", "glass", "metal"],
            "lighting": ["studio", "dramatic"],
            "moods": ["modern", "minimal"],
            "props": ["laptop", "smartphone", "cables"]
        },
        "fashion": {
            "surfaces": ["fabric", "marble", "wood"],
            "lighting": ["natural", "soft"],
            "moods": ["luxury", "vintage"],
            "props": ["accessories", "perfume", "flowers"]
        }
    }
    
    def __init__(self, model_type: str = "sdxl", float16: bool = True):
        # Initialize base models
        self.base_model = StableDiffusionXL(
            model="stabilityai/sdxl-turbo",
            float16=float16
        )
        print("Loading SDXL model for AGC Ads...")
        self.base_model.ensure_models_are_loaded()
        
        # Initialize pipelines
        self.corepulse = CorePulseMLX(self.base_model)
        self.hallucination_free = HallucinationFreeProductPlacement(
            model_type=model_type,
            float16=float16
        )
        
        print("AGC Ads Engine initialized!")
        
        # Performance tracking
        self.metrics = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "avg_generation_time": 0,
            "conversion_predictions": {}
        }
    
    def generate_campaign(
        self,
        campaign: AdCampaign,
        products: List[Dict[str, str]],
        formats: List[str],
        variants_per_ad: int = 3,
        batch_size: int = 4
    ) -> List[ProductAd]:
        """
        Generate full ad campaign with multiple products and formats
        
        Args:
            campaign: Campaign configuration
            products: List of product configs [{image, name, description}]
            formats: List of format names from AD_FORMATS
            variants_per_ad: Number of A/B test variants
            batch_size: Parallel processing batch size
        """
        print(f"\n{'='*60}")
        print(f"AGC ADS CAMPAIGN: {campaign.campaign_id}")
        print(f"{'='*60}")
        print(f"Brand: {campaign.brand_name}")
        print(f"Products: {len(products)}")
        print(f"Formats: {formats}")
        print(f"Total ads to generate: {len(products) * len(formats) * variants_per_ad}")
        
        all_ads = []
        
        # Process in batches for efficiency
        for product in products:
            for format_name in formats:
                format_spec = self.AD_FORMATS[format_name]
                
                # Generate variants for A/B testing
                for variant_idx in range(variants_per_ad):
                    ad = self._generate_single_ad(
                        campaign=campaign,
                        product=product,
                        format_spec=format_spec,
                        variant_index=variant_idx
                    )
                    all_ads.append(ad)
                    
                    # Track metrics
                    self.metrics["total_generated"] += 1
                    if ad.performance_metrics.get("generated", False):
                        self.metrics["successful"] += 1
                    else:
                        self.metrics["failed"] += 1
        
        return all_ads
    
    def _generate_single_ad(
        self,
        campaign: AdCampaign,
        product: Dict[str, str],
        format_spec: AdFormat,
        variant_index: int
    ) -> ProductAd:
        """Generate single ad with hallucination-free placement"""
        
        # Create unique ad ID
        ad_id = f"{campaign.campaign_id}_{product['name']}_{format_spec.name}_{variant_index}"
        ad_id = hashlib.md5(ad_id.encode()).hexdigest()[:12]
        
        # Select scene based on product category
        scene_config = self._select_scene_config(
            campaign.product_category,
            variant_index
        )
        
        # Generate headlines and CTA
        headlines = self._generate_headlines(
            product=product,
            brand_voice=campaign.brand_voice,
            variant=variant_index
        )
        
        # Create the ad
        print(f"\nGenerating Ad: {ad_id}")
        print(f"  Product: {product['name']}")
        print(f"  Format: {format_spec.name} ({format_spec.width}x{format_spec.height})")
        print(f"  Variant: {variant_index + 1}")
        
        # Generate base scene with CorePulse control
        scene_prompt = self._build_scene_prompt(
            scene_config=scene_config,
            brand_voice=campaign.brand_voice,
            format_spec=format_spec
        )
        
        # Multi-level injections for brand consistency
        injections = [
            # Brand identity (early)
            PromptInjection(
                prompt=f"{campaign.brand_name} brand aesthetic, {campaign.brand_voice} style",
                levels=[InjectionLevel.ENCODER_EARLY],
                strength=0.9
            ),
            # Scene composition (mid)
            PromptInjection(
                prompt=scene_prompt,
                levels=[InjectionLevel.ENCODER_MID, InjectionLevel.DECODER_MID],
                strength=0.8
            ),
            # Lighting and mood (late)
            PromptInjection(
                prompt=f"{scene_config['lighting']} lighting, {scene_config['mood']} atmosphere",
                levels=[InjectionLevel.DECODER_LATE],
                strength=0.7
            )
        ]
        
        # Token emphasis for brand elements
        token_masks = [
            TokenMask(
                tokens=[campaign.brand_name, campaign.product_category],
                mask_type="amplify",
                strength=1.5
            )
        ]
        
        # Generate scene
        output_path = f"agc_ads/{ad_id}.png"
        Path("agc_ads").mkdir(exist_ok=True)
        
        try:
            # Use hallucination-free placement
            result = self._place_product_in_scene(
                product_image=product['image'],
                scene_prompt=scene_prompt,
                injections=injections,
                token_masks=token_masks,
                output_size=(format_spec.width, format_spec.height),
                output_path=output_path
            )
            
            # Add text overlays and branding
            final_ad = self._add_branding_and_text(
                base_image=result,
                headlines=headlines,
                campaign=campaign,
                format_spec=format_spec
            )
            
            # Save final ad
            final_path = f"agc_ads/{ad_id}_final.png"
            final_ad.save(final_path)
            
            # Predict performance
            performance = self._predict_performance(
                ad_image=final_ad,
                scene_config=scene_config,
                headlines=headlines
            )
            
            generated = True
            
        except Exception as e:
            print(f"  ⚠️ Error generating ad: {e}")
            generated = False
            performance = {"error": str(e)}
            final_path = None
        
        # Create ad object
        ad = ProductAd(
            ad_id=ad_id,
            campaign=campaign,
            product_image=product['image'],
            headline=headlines['headline'],
            subheadline=headlines.get('subheadline'),
            cta_text=headlines['cta'],
            scene_theme=scene_config['theme'],
            format=format_spec,
            variants=[f"variant_{variant_index}"],
            performance_metrics={
                "generated": generated,
                "path": final_path,
                "predicted_ctr": performance.get("ctr", 0),
                "predicted_conversion": performance.get("conversion", 0),
                "quality_score": performance.get("quality", 0)
            }
        )
        
        return ad
    
    def _place_product_in_scene(
        self,
        product_image: str,
        scene_prompt: str,
        injections: List[PromptInjection],
        token_masks: List[TokenMask],
        output_size: Tuple[int, int],
        output_path: str
    ) -> Image.Image:
        """Place product with zero hallucination"""
        
        # Generate scene with CorePulse
        result = self.corepulse.generate_with_control(
            base_prompt=scene_prompt,
            prompt_injections=injections,
            token_masks=token_masks,
            negative_prompt="blurry, distorted, text, watermark, logo",
            num_steps=4,  # SDXL Turbo
            cfg_weight=0.0,
            output_size=output_size
        )
        
        # Convert to PIL
        img_array = np.array(result)
        if img_array.ndim == 4:
            img_array = img_array[0]
        if img_array.shape[0] == 3:
            img_array = np.transpose(img_array, (1, 2, 0))
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        scene = Image.fromarray(img_array).resize(output_size, Image.Resampling.LANCZOS)
        
        # Extract product with alpha
        product = Image.open(product_image).convert("RGBA")
        
        # Smart placement based on format
        if output_size[1] > output_size[0]:  # Portrait
            scale = 0.4
            position = (
                (output_size[0] - int(product.width * scale)) // 2,
                int(output_size[1] * 0.4)
            )
        else:  # Landscape or square
            scale = 0.5
            position = (
                (output_size[0] - int(product.width * scale)) // 2,
                (output_size[1] - int(product.height * scale)) // 2
            )
        
        # Resize product
        new_size = (int(product.width * scale), int(product.height * scale))
        product = product.resize(new_size, Image.Resampling.LANCZOS)
        
        # Add shadow
        shadow = self._create_ad_shadow(product, scene.size, position)
        scene = Image.alpha_composite(scene.convert('RGBA'), shadow).convert('RGB')
        
        # Place product (zero hallucination)
        scene.paste(product, position, product.split()[3] if product.mode == 'RGBA' else None)
        
        return scene
    
    def _create_ad_shadow(
        self,
        product: Image.Image,
        canvas_size: Tuple[int, int],
        position: Tuple[int, int]
    ) -> Image.Image:
        """Create professional ad shadow"""
        shadow = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        
        # Extract alpha or create from product
        if product.mode == 'RGBA':
            _, _, _, alpha = product.split()
        else:
            alpha = Image.new('L', product.size, 255)
        
        # Multi-layer shadow for depth
        for offset, blur, opacity in [(5, 10, 80), (10, 20, 60), (20, 40, 40)]:
            layer = Image.new('L', canvas_size, 0)
            shadow_pos = (position[0] + offset, position[1] + offset)
            layer.paste(alpha, shadow_pos)
            layer = layer.filter(ImageFilter.GaussianBlur(radius=blur))
            
            shadow_layer = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
            shadow_layer.paste((0, 0, 0, opacity), mask=layer)
            shadow = Image.alpha_composite(shadow, shadow_layer)
        
        return shadow
    
    def _add_branding_and_text(
        self,
        base_image: Image.Image,
        headlines: Dict[str, str],
        campaign: AdCampaign,
        format_spec: AdFormat
    ) -> Image.Image:
        """Add text overlays, CTA buttons, and branding"""
        
        img = base_image.copy()
        draw = ImageDraw.Draw(img)
        
        # Try to load fonts (fallback to default if not available)
        try:
            headline_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            subheadline_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
            cta_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        except:
            headline_font = ImageFont.load_default()
            subheadline_font = ImageFont.load_default()
            cta_font = ImageFont.load_default()
        
        # Add headline
        headline_y = 50 if format_spec.platform != "instagram_story" else 100
        draw.text(
            (format_spec.width // 2, headline_y),
            headlines['headline'],
            fill=(255, 255, 255),
            font=headline_font,
            anchor="mt",
            stroke_width=2,
            stroke_fill=(0, 0, 0)
        )
        
        # Add subheadline if exists
        if headlines.get('subheadline'):
            draw.text(
                (format_spec.width // 2, headline_y + 60),
                headlines['subheadline'],
                fill=(220, 220, 220),
                font=subheadline_font,
                anchor="mt"
            )
        
        # Add CTA button
        cta_padding = 20
        cta_text = headlines['cta']
        
        # Calculate CTA button position
        if format_spec.cta_position == "bottom_center":
            cta_x = format_spec.width // 2
            cta_y = format_spec.height - 80
        else:  # bottom_right
            cta_x = format_spec.width - 150
            cta_y = format_spec.height - 80
        
        # Draw CTA button
        button_width = 200
        button_height = 60
        button_rect = [
            cta_x - button_width // 2,
            cta_y - button_height // 2,
            cta_x + button_width // 2,
            cta_y + button_height // 2
        ]
        
        # Button with brand color
        brand_color = self._parse_brand_color(campaign.brand_colors[0] if campaign.brand_colors else "#FF6B6B")
        draw.rounded_rectangle(button_rect, radius=30, fill=brand_color)
        
        # CTA text
        draw.text(
            (cta_x, cta_y),
            cta_text,
            fill=(255, 255, 255),
            font=cta_font,
            anchor="mm"
        )
        
        # Add brand watermark (subtle)
        watermark_text = f"© {campaign.brand_name}"
        draw.text(
            (format_spec.width - 10, format_spec.height - 10),
            watermark_text,
            fill=(200, 200, 200, 128),
            anchor="rb"
        )
        
        return img
    
    def _select_scene_config(
        self,
        product_category: str,
        variant_index: int
    ) -> Dict:
        """Select scene configuration based on product and variant"""
        
        # Map category to theme
        category_themes = {
            "electronics": "tech",
            "fashion": "fashion",
            "jewelry": "luxury",
            "beauty": "fashion",
            "home": "lifestyle",
            "sports": "lifestyle"
        }
        
        theme = category_themes.get(product_category.lower(), "lifestyle")
        scene_options = self.CONVERSION_OPTIMIZED_SCENES[theme]
        
        # Rotate through options for variants
        config = {
            "theme": theme,
            "surface": scene_options["surfaces"][variant_index % len(scene_options["surfaces"])],
            "lighting": scene_options["lighting"][variant_index % len(scene_options["lighting"])],
            "mood": scene_options["moods"][variant_index % len(scene_options["moods"])],
            "props": scene_options["props"]
        }
        
        return config
    
    def _generate_headlines(
        self,
        product: Dict,
        brand_voice: str,
        variant: int
    ) -> Dict[str, str]:
        """Generate conversion-optimized headlines"""
        
        # Templates based on brand voice
        templates = {
            "luxury": {
                "headlines": [
                    f"Elevate Your {product['name']}",
                    f"Exclusively Crafted {product['name']}",
                    f"Redefine Elegance with {product['name']}"
                ],
                "subheadlines": [
                    "Timeless sophistication",
                    "Uncompromising quality",
                    "Where luxury meets innovation"
                ],
                "ctas": ["Discover More", "Experience Luxury", "Shop Exclusive"]
            },
            "casual": {
                "headlines": [
                    f"Meet Your New Favorite {product['name']}",
                    f"{product['name']} That Just Gets It",
                    f"Say Hello to {product['name']}"
                ],
                "subheadlines": [
                    "Made for real life",
                    "Simple. Beautiful. Yours.",
                    "Because you deserve better"
                ],
                "ctas": ["Get Yours", "Shop Now", "Learn More"]
            },
            "professional": {
                "headlines": [
                    f"Professional-Grade {product['name']}",
                    f"Optimize with {product['name']}",
                    f"Enterprise {product['name']} Solutions"
                ],
                "subheadlines": [
                    "Trusted by industry leaders",
                    "Performance that delivers",
                    "Built for professionals"
                ],
                "ctas": ["Get Started", "Request Demo", "Learn More"]
            },
            "playful": {
                "headlines": [
                    f"OMG! {product['name']} is Here!",
                    f"{product['name']} FTW! 🎉",
                    f"Your {product['name']} Glow-Up"
                ],
                "subheadlines": [
                    "Join the fun!",
                    "Limited time vibes",
                    "You + this = perfect"
                ],
                "ctas": ["Grab It!", "Yes Please!", "I Want This!"]
            }
        }
        
        voice_templates = templates.get(brand_voice, templates["casual"])
        
        return {
            "headline": voice_templates["headlines"][variant % len(voice_templates["headlines"])],
            "subheadline": voice_templates["subheadlines"][variant % len(voice_templates["subheadlines"])],
            "cta": voice_templates["ctas"][variant % len(voice_templates["ctas"])]
        }
    
    def _build_scene_prompt(
        self,
        scene_config: Dict,
        brand_voice: str,
        format_spec: AdFormat
    ) -> str:
        """Build optimized scene prompt"""
        
        # Base elements
        elements = [
            f"{scene_config['surface']} surface",
            f"{scene_config['lighting']} lighting",
            f"{scene_config['mood']} atmosphere"
        ]
        
        # Add props
        if scene_config.get('props'):
            elements.append(f"with {', '.join(scene_config['props'][:2])}")
        
        # Platform-specific optimizations
        if format_spec.platform == "instagram":
            elements.append("instagram-worthy aesthetic")
        elif format_spec.platform == "youtube":
            elements.append("eye-catching thumbnail style")
        elif format_spec.platform == "linkedin":
            elements.append("professional business setting")
        
        return ", ".join(elements) + ", high-end product photography"
    
    def _predict_performance(
        self,
        ad_image: Image.Image,
        scene_config: Dict,
        headlines: Dict
    ) -> Dict[str, float]:
        """Predict ad performance metrics"""
        
        # Simplified performance prediction based on best practices
        score = 70.0  # Base score
        
        # Scene quality factors
        if scene_config['theme'] == 'luxury':
            score += 10
        if scene_config['lighting'] in ['natural', 'soft']:
            score += 5
        
        # Headline factors
        if len(headlines['headline']) < 30:
            score += 5  # Concise is better
        if headlines['cta'] in ['Shop Now', 'Get Yours', 'Learn More']:
            score += 10  # Proven CTAs
        
        # Visual factors (simplified)
        img_array = np.array(ad_image)
        brightness = np.mean(img_array)
        if 100 < brightness < 200:
            score += 5  # Good contrast
        
        # Normalize score
        score = min(100, max(0, score))
        
        return {
            "quality": score,
            "ctr": score * 0.05,  # Estimated CTR %
            "conversion": score * 0.02  # Estimated conversion %
        }
    
    def _parse_brand_color(self, color_str: str) -> Tuple[int, int, int]:
        """Parse brand color from hex or name"""
        if color_str.startswith("#"):
            # Parse hex
            hex_color = color_str.lstrip("#")
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        else:
            # Default colors
            colors = {
                "red": (255, 67, 67),
                "blue": (67, 133, 255),
                "green": (67, 255, 133),
                "purple": (133, 67, 255),
                "orange": (255, 133, 67)
            }
            return colors.get(color_str.lower(), (255, 67, 67))
    
    def export_campaign_report(
        self,
        ads: List[ProductAd],
        output_path: str = "campaign_report.json"
    ) -> Dict:
        """Export campaign report with all metrics"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_ads": len(ads),
                "successful": sum(1 for ad in ads if ad.performance_metrics.get("generated")),
                "failed": sum(1 for ad in ads if not ad.performance_metrics.get("generated")),
                "avg_quality_score": np.mean([
                    ad.performance_metrics.get("quality_score", 0)
                    for ad in ads
                ]),
                "avg_predicted_ctr": np.mean([
                    ad.performance_metrics.get("predicted_ctr", 0)
                    for ad in ads
                ]),
                "avg_predicted_conversion": np.mean([
                    ad.performance_metrics.get("predicted_conversion", 0)
                    for ad in ads
                ])
            },
            "by_format": {},
            "by_variant": {},
            "ads": []
        }
        
        # Group metrics by format
        for format_name in set(ad.format.name for ad in ads):
            format_ads = [ad for ad in ads if ad.format.name == format_name]
            report["by_format"][format_name] = {
                "count": len(format_ads),
                "avg_quality": np.mean([
                    ad.performance_metrics.get("quality_score", 0)
                    for ad in format_ads
                ])
            }
        
        # Include ad details
        for ad in ads:
            report["ads"].append({
                "ad_id": ad.ad_id,
                "product": ad.product_image,
                "format": ad.format.name,
                "headline": ad.headline,
                "cta": ad.cta_text,
                "metrics": ad.performance_metrics
            })
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Campaign report saved to: {output_path}")
        
        return report


def main():
    """Demo AGC Ads Pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AGC Ads Production Pipeline")
    parser.add_argument("--campaign", default="DEMO2024", help="Campaign ID")
    parser.add_argument("--brand", default="TechLux", help="Brand name")
    parser.add_argument("--products", nargs="+", default=["test_product_watch.png"], help="Product images")
    parser.add_argument("--formats", nargs="+", default=["instagram_feed"], help="Ad formats")
    parser.add_argument("--variants", type=int, default=3, help="Variants per ad")
    parser.add_argument("--voice", default="luxury", choices=["luxury", "casual", "professional", "playful"])
    
    args = parser.parse_args()
    
    # Create campaign
    campaign = AdCampaign(
        campaign_id=args.campaign,
        brand_name=args.brand,
        product_category="electronics",
        target_audience="25-45 professionals",
        brand_colors=["#FF6B6B", "#4A90E2"],
        brand_voice=args.voice
    )
    
    # Prepare products
    products = [
        {
            "image": prod,
            "name": Path(prod).stem.replace("_", " ").title(),
            "description": "Premium product"
        }
        for prod in args.products
    ]
    
    # Initialize engine
    engine = AGCAdsEngine()
    
    # Generate campaign
    ads = engine.generate_campaign(
        campaign=campaign,
        products=products,
        formats=args.formats,
        variants_per_ad=args.variants
    )
    
    # Export report
    report = engine.export_campaign_report(ads, f"agc_campaign_{campaign.campaign_id}.json")
    
    print("\n" + "="*60)
    print("AGC ADS CAMPAIGN COMPLETE")
    print("="*60)
    print(f"✅ Generated {report['summary']['successful']} ads")
    print(f"📊 Avg Quality Score: {report['summary']['avg_quality_score']:.1f}/100")
    print(f"🎯 Avg Predicted CTR: {report['summary']['avg_predicted_ctr']:.2f}%")
    print(f"💰 Avg Predicted Conversion: {report['summary']['avg_predicted_conversion']:.2f}%")


if __name__ == "__main__":
    main()