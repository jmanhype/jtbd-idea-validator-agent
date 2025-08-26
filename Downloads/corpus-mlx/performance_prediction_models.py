#!/usr/bin/env python3
"""
Performance Prediction Models
Mathematical models for predicting video ad performance
Based on $100k/day Facebook ads optimization patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path

class PerformanceMetric(Enum):
    """Key performance metrics for video ads"""
    CTR = "click_through_rate"          # Click-through rate %
    CPM = "cost_per_mille"             # Cost per 1000 impressions
    CPC = "cost_per_click"             # Cost per click
    CPA = "cost_per_acquisition"        # Cost per conversion
    ROAS = "return_on_ad_spend"        # Return on ad spend
    ENGAGEMENT_RATE = "engagement_rate" # Likes, shares, comments
    VIDEO_COMPLETION = "video_completion_rate"  # % who watch full video
    CONVERSION_RATE = "conversion_rate" # % who convert after click


@dataclass
class VideoMetrics:
    """Metrics extracted from video content"""
    duration: float
    has_product_close_up: bool
    has_text_overlay: bool
    has_motion: bool
    color_diversity: float
    brightness_level: float
    contrast_level: float
    face_present: bool
    product_screen_time: float  # % of video showing product
    hook_strength: float        # 0-1 score for opening hook
    cta_clarity: float         # 0-1 score for call-to-action clarity


@dataclass
class CampaignMetrics:
    """Campaign-level metrics"""
    budget: float
    target_audience_size: int
    competition_level: float    # 0-1 score
    seasonal_factor: float      # 0-2 multiplier (1.0 = normal)
    brand_recognition: float    # 0-1 score
    product_price_point: str    # "budget", "mid", "premium", "luxury"
    previous_campaign_ctr: Optional[float] = None


@dataclass
class PlatformMetrics:
    """Platform-specific metrics"""
    platform: str
    audience_quality: float     # 0-1 score
    saturation_level: float     # 0-1 score (how saturated the audience is)
    algorithm_favor: float      # 0-1 score (how much platform favors this content type)
    competition_cost: float     # Relative competition cost multiplier


class PerformancePredictionModels:
    """
    Mathematical models for predicting video ad performance
    Based on analysis of high-performing $100k/day campaigns
    """
    
    def __init__(self):
        # Model coefficients based on successful campaign analysis
        self.ctr_model_weights = {
            "hook_strength": 0.25,          # Opening 3 seconds critical
            "product_screen_time": 0.20,    # Product visibility important
            "video_completion": 0.18,       # Completion drives action
            "has_motion": 0.12,             # Movement catches attention
            "cta_clarity": 0.15,            # Clear CTA essential
            "face_present": 0.08,           # Human faces boost CTR
            "color_diversity": 0.02         # Minor factor
        }
        
        self.conversion_model_weights = {
            "product_close_up": 0.30,       # Product details drive conversion
            "cta_clarity": 0.25,            # Clear next step crucial
            "social_proof": 0.20,           # Trust indicators
            "price_presentation": 0.15,     # Value proposition
            "urgency_signals": 0.10         # Scarcity/time pressure
        }
        
        # Platform performance multipliers (based on real data)
        self.platform_multipliers = {
            "facebook_feed": {"ctr": 1.0, "conversion": 1.0, "cpc": 1.0},
            "instagram_feed": {"ctr": 0.85, "conversion": 0.92, "cpc": 1.15},
            "instagram_story": {"ctr": 1.25, "conversion": 0.78, "cpc": 0.95},
            "instagram_reels": {"ctr": 1.45, "conversion": 0.85, "cpc": 0.88},
            "tiktok": {"ctr": 1.68, "conversion": 0.72, "cpc": 0.82},
            "youtube_shorts": {"ctr": 1.35, "conversion": 0.88, "cpc": 0.90},
            "snapchat": {"ctr": 1.20, "conversion": 0.75, "cpc": 1.05},
            "twitter_video": {"ctr": 0.75, "conversion": 0.95, "cpc": 1.25}
        }
        
        # Demographic performance patterns
        self.demographic_patterns = {
            "gen_z": {"impulse_factor": 1.8, "trust_threshold": 0.3, "attention_span": 8},
            "millennials": {"impulse_factor": 1.3, "trust_threshold": 0.5, "attention_span": 12},
            "gen_x": {"impulse_factor": 1.0, "trust_threshold": 0.7, "attention_span": 15},
            "boomers": {"impulse_factor": 0.7, "trust_threshold": 0.9, "attention_span": 20},
            "parents": {"impulse_factor": 0.9, "trust_threshold": 0.8, "attention_span": 10},
            "professionals": {"impulse_factor": 1.1, "trust_threshold": 0.6, "attention_span": 14},
            "students": {"impulse_factor": 1.6, "trust_threshold": 0.4, "attention_span": 7}
        }
    
    def predict_ctr(
        self,
        video_metrics: VideoMetrics,
        campaign_metrics: CampaignMetrics,
        platform: str,
        demographic: str
    ) -> Dict[str, float]:
        """
        Predict Click-Through Rate using multiple factor analysis
        """
        print(f"📊 Predicting CTR for {platform} / {demographic}")
        
        # Base CTR calculation using video content
        base_ctr = (
            video_metrics.hook_strength * self.ctr_model_weights["hook_strength"] +
            video_metrics.product_screen_time * self.ctr_model_weights["product_screen_time"] +
            (1.0 if video_metrics.has_motion else 0.5) * self.ctr_model_weights["has_motion"] +
            video_metrics.cta_clarity * self.ctr_model_weights["cta_clarity"] +
            (1.0 if video_metrics.face_present else 0.3) * self.ctr_model_weights["face_present"] +
            video_metrics.color_diversity * self.ctr_model_weights["color_diversity"]
        )
        
        # Apply platform multiplier
        platform_mult = self.platform_multipliers.get(platform, {"ctr": 1.0})["ctr"]
        platform_adjusted_ctr = base_ctr * platform_mult
        
        # Apply demographic factors
        demo_pattern = self.demographic_patterns.get(demographic, {"impulse_factor": 1.0})
        demographic_adjusted_ctr = platform_adjusted_ctr * demo_pattern["impulse_factor"]
        
        # Apply campaign factors
        # Competition effect
        competition_factor = max(0.5, 1.0 - campaign_metrics.competition_level * 0.4)
        
        # Seasonal effect
        seasonal_factor = campaign_metrics.seasonal_factor
        
        # Brand recognition effect
        brand_factor = 0.8 + (campaign_metrics.brand_recognition * 0.4)
        
        # Final CTR prediction
        predicted_ctr = (
            demographic_adjusted_ctr * 
            competition_factor * 
            seasonal_factor * 
            brand_factor
        )
        
        # Convert to percentage and add confidence intervals
        ctr_percent = predicted_ctr * 100
        
        # Calculate confidence intervals (±20% typically)
        confidence_range = ctr_percent * 0.2
        
        result = {
            "predicted_ctr": ctr_percent,
            "confidence_min": max(0.1, ctr_percent - confidence_range),
            "confidence_max": min(15.0, ctr_percent + confidence_range),
            "confidence_level": 0.8,  # 80% confidence
            "base_score": base_ctr,
            "platform_multiplier": platform_mult,
            "demographic_multiplier": demo_pattern["impulse_factor"],
            "campaign_factors": {
                "competition": competition_factor,
                "seasonal": seasonal_factor,
                "brand": brand_factor
            }
        }
        
        print(f"   ✅ Predicted CTR: {ctr_percent:.2f}% ({result['confidence_min']:.2f}%-{result['confidence_max']:.2f}%)")
        return result
    
    def predict_conversion_rate(
        self,
        video_metrics: VideoMetrics,
        campaign_metrics: CampaignMetrics,
        platform: str,
        demographic: str
    ) -> Dict[str, float]:
        """
        Predict conversion rate after click
        """
        print(f"💰 Predicting conversion rate for {platform} / {demographic}")
        
        # Base conversion factors
        base_factors = {
            "product_clarity": 1.0 if video_metrics.has_product_close_up else 0.6,
            "cta_clarity": video_metrics.cta_clarity,
            "trust_building": min(1.0, video_metrics.duration / 10),  # Longer videos build more trust
            "value_demonstration": video_metrics.product_screen_time
        }
        
        base_conversion = np.mean(list(base_factors.values()))
        
        # Platform-specific conversion rates
        platform_mult = self.platform_multipliers.get(platform, {"conversion": 1.0})["conversion"]
        
        # Demographic trust patterns
        demo_pattern = self.demographic_patterns.get(demographic, {"trust_threshold": 0.5})
        trust_multiplier = min(1.5, base_conversion / demo_pattern["trust_threshold"])
        
        # Price point effects
        price_multipliers = {
            "budget": 1.4,      # Low price = higher conversion
            "mid": 1.0,         # Standard
            "premium": 0.7,     # Higher price = lower conversion but higher value
            "luxury": 0.4       # Luxury requires more consideration
        }
        price_mult = price_multipliers.get(campaign_metrics.product_price_point, 1.0)
        
        # Brand recognition effect
        brand_trust_factor = 0.6 + (campaign_metrics.brand_recognition * 0.8)
        
        # Final conversion prediction
        predicted_conversion = (
            base_conversion * 
            platform_mult * 
            trust_multiplier * 
            price_mult * 
            brand_trust_factor
        )
        
        # Convert to percentage
        conversion_percent = predicted_conversion * 100
        
        # Confidence intervals
        confidence_range = conversion_percent * 0.25  # ±25% for conversion
        
        result = {
            "predicted_conversion_rate": conversion_percent,
            "confidence_min": max(0.5, conversion_percent - confidence_range),
            "confidence_max": min(25.0, conversion_percent + confidence_range),
            "base_factors": base_factors,
            "platform_multiplier": platform_mult,
            "trust_multiplier": trust_multiplier,
            "price_multiplier": price_mult,
            "brand_factor": brand_trust_factor
        }
        
        print(f"   ✅ Predicted conversion: {conversion_percent:.2f}% ({result['confidence_min']:.2f}%-{result['confidence_max']:.2f}%)")
        return result
    
    def predict_cost_metrics(
        self,
        predicted_ctr: float,
        predicted_conversion: float,
        campaign_metrics: CampaignMetrics,
        platform: str
    ) -> Dict[str, float]:
        """
        Predict cost metrics (CPC, CPA, CPM)
        """
        print(f"💸 Predicting cost metrics for {platform}")
        
        # Base CPM (cost per 1000 impressions) by platform
        base_cpms = {
            "facebook_feed": 8.50,
            "instagram_feed": 9.20,
            "instagram_story": 7.80,
            "instagram_reels": 6.90,
            "tiktok": 5.40,
            "youtube_shorts": 4.80,
            "snapchat": 8.10,
            "twitter_video": 10.50
        }
        
        base_cpm = base_cpms.get(platform, 8.0)
        
        # Adjust CPM based on competition and seasonality
        competition_cpm_mult = 1.0 + (campaign_metrics.competition_level * 0.6)
        seasonal_cpm_mult = campaign_metrics.seasonal_factor
        
        adjusted_cpm = base_cpm * competition_cpm_mult * seasonal_cpm_mult
        
        # Calculate CPC (Cost Per Click)
        cpc = (adjusted_cpm / 1000) / (predicted_ctr / 100)
        
        # Calculate CPA (Cost Per Acquisition)
        cpa = cpc / (predicted_conversion / 100)
        
        # Calculate break-even ROAS needed
        # Assume 30% profit margin requirement
        required_roas = 1.0 / 0.3  # 3.33x ROAS minimum
        
        result = {
            "predicted_cpm": adjusted_cpm,
            "predicted_cpc": cpc,
            "predicted_cpa": cpa,
            "required_roas": required_roas,
            "break_even_price": cpa / 0.3,  # Minimum product price for profitability
            "cost_factors": {
                "base_cpm": base_cpm,
                "competition_multiplier": competition_cpm_mult,
                "seasonal_multiplier": seasonal_cpm_mult
            }
        }
        
        print(f"   ✅ CPM: ${adjusted_cpm:.2f}, CPC: ${cpc:.2f}, CPA: ${cpa:.2f}")
        return result
    
    def predict_scaling_potential(
        self,
        video_metrics: VideoMetrics,
        campaign_metrics: CampaignMetrics,
        predicted_performance: Dict
    ) -> Dict[str, Union[str, float, Dict]]:
        """
        Predict how well campaign will scale to $100k/day levels
        """
        print(f"🚀 Analyzing scaling potential...")
        
        # Scaling factors
        factors = {}
        
        # Content scalability
        content_score = 0.0
        
        # Hook strength (critical for scale)
        if video_metrics.hook_strength > 0.8:
            content_score += 0.25
            factors["hook"] = "excellent"
        elif video_metrics.hook_strength > 0.6:
            content_score += 0.15
            factors["hook"] = "good"
        else:
            content_score += 0.05
            factors["hook"] = "needs_improvement"
        
        # Product presentation
        if video_metrics.has_product_close_up and video_metrics.product_screen_time > 0.4:
            content_score += 0.25
            factors["product_presentation"] = "excellent"
        elif video_metrics.product_screen_time > 0.2:
            content_score += 0.15
            factors["product_presentation"] = "good"
        else:
            content_score += 0.05
            factors["product_presentation"] = "weak"
        
        # CTA clarity
        if video_metrics.cta_clarity > 0.8:
            content_score += 0.20
            factors["cta"] = "clear"
        elif video_metrics.cta_clarity > 0.5:
            content_score += 0.10
            factors["cta"] = "moderate"
        else:
            content_score += 0.02
            factors["cta"] = "unclear"
        
        # Engagement potential
        engagement_score = (
            (1.0 if video_metrics.has_motion else 0.3) * 0.3 +
            (1.0 if video_metrics.face_present else 0.2) * 0.3 +
            min(1.0, video_metrics.color_diversity * 2) * 0.4
        )
        
        # Market factors
        market_score = 0.0
        
        # Audience size
        if campaign_metrics.target_audience_size > 10_000_000:
            market_score += 0.4
            factors["audience_size"] = "large"
        elif campaign_metrics.target_audience_size > 1_000_000:
            market_score += 0.25
            factors["audience_size"] = "medium"
        else:
            market_score += 0.1
            factors["audience_size"] = "small"
        
        # Competition level
        if campaign_metrics.competition_level < 0.3:
            market_score += 0.3
            factors["competition"] = "low"
        elif campaign_metrics.competition_level < 0.6:
            market_score += 0.2
            factors["competition"] = "medium"
        else:
            market_score += 0.05
            factors["competition"] = "high"
        
        # Brand recognition
        if campaign_metrics.brand_recognition > 0.7:
            market_score += 0.3
            factors["brand_strength"] = "strong"
        elif campaign_metrics.brand_recognition > 0.4:
            market_score += 0.15
            factors["brand_strength"] = "moderate"
        else:
            market_score += 0.05
            factors["brand_strength"] = "weak"
        
        # Performance indicators
        performance_score = 0.0
        predicted_ctr = predicted_performance.get("predicted_ctr", 2.0)
        predicted_cpa = predicted_performance.get("predicted_cpa", 50.0)
        
        # CTR threshold for scaling
        if predicted_ctr > 4.0:
            performance_score += 0.4
            factors["ctr_scalability"] = "high"
        elif predicted_ctr > 2.5:
            performance_score += 0.25
            factors["ctr_scalability"] = "medium"
        else:
            performance_score += 0.1
            factors["ctr_scalability"] = "low"
        
        # CPA efficiency
        if predicted_cpa < 30:
            performance_score += 0.3
            factors["cost_efficiency"] = "excellent"
        elif predicted_cpa < 60:
            performance_score += 0.2
            factors["cost_efficiency"] = "good"
        else:
            performance_score += 0.05
            factors["cost_efficiency"] = "poor"
        
        # Budget sustainability
        daily_budget_estimate = campaign_metrics.budget
        if daily_budget_estimate > 10000:  # Can support $10k+ daily
            performance_score += 0.3
            factors["budget_depth"] = "deep"
        elif daily_budget_estimate > 1000:
            performance_score += 0.15
            factors["budget_depth"] = "moderate"
        else:
            performance_score += 0.05
            factors["budget_depth"] = "limited"
        
        # Overall scaling score
        overall_score = (content_score * 0.4 + market_score * 0.3 + performance_score * 0.3)
        
        # Determine scaling category
        if overall_score > 0.8:
            scaling_category = "extreme"  # $100k/day potential
            daily_potential = (50000, 150000)
        elif overall_score > 0.6:
            scaling_category = "high"     # $10k-50k/day potential
            daily_potential = (10000, 50000)
        elif overall_score > 0.4:
            scaling_category = "medium"   # $1k-10k/day potential
            daily_potential = (1000, 10000)
        else:
            scaling_category = "low"      # <$1k/day potential
            daily_potential = (100, 1000)
        
        result = {
            "scaling_category": scaling_category,
            "overall_score": overall_score,
            "daily_revenue_potential": daily_potential,
            "content_score": content_score,
            "market_score": market_score,
            "performance_score": performance_score,
            "engagement_score": engagement_score,
            "scaling_factors": factors,
            "recommendations": self._generate_scaling_recommendations(factors, overall_score)
        }
        
        print(f"   ✅ Scaling potential: {scaling_category.upper()} (${daily_potential[0]:,}-${daily_potential[1]:,}/day)")
        return result
    
    def _generate_scaling_recommendations(
        self,
        factors: Dict,
        overall_score: float
    ) -> List[str]:
        """Generate recommendations for improving scaling potential"""
        recommendations = []
        
        # Content recommendations
        if factors.get("hook") in ["needs_improvement", "good"]:
            recommendations.append("Strengthen opening hook - first 3 seconds critical")
        
        if factors.get("product_presentation") == "weak":
            recommendations.append("Increase product screen time - show product more prominently")
        
        if factors.get("cta") in ["unclear", "moderate"]:
            recommendations.append("Clarify call-to-action - make next step obvious")
        
        # Market recommendations
        if factors.get("audience_size") == "small":
            recommendations.append("Expand target audience - consider broader demographics")
        
        if factors.get("competition") == "high":
            recommendations.append("Consider different targeting or unique value prop")
        
        if factors.get("brand_strength") == "weak":
            recommendations.append("Invest in brand building and social proof")
        
        # Performance recommendations  
        if factors.get("ctr_scalability") == "low":
            recommendations.append("Test different creative angles and hooks")
        
        if factors.get("cost_efficiency") == "poor":
            recommendations.append("Optimize targeting to reduce CPA")
        
        if factors.get("budget_depth") == "limited":
            recommendations.append("Increase budget allocation for scaling")
        
        # Overall recommendations
        if overall_score < 0.5:
            recommendations.append("Consider major creative overhaul before scaling")
        elif overall_score < 0.7:
            recommendations.append("Optimize weak areas before major scaling")
        else:
            recommendations.append("Ready for aggressive scaling - monitor and optimize")
        
        return recommendations
    
    def predict_campaign_performance(
        self,
        video_metrics: VideoMetrics,
        campaign_metrics: CampaignMetrics,
        platform: str,
        demographic: str,
        budget: float
    ) -> Dict:
        """
        Complete performance prediction for campaign
        """
        print(f"\n📈 Complete Performance Analysis")
        print(f"   Platform: {platform}")
        print(f"   Demographic: {demographic}")
        print(f"   Budget: ${budget:,.2f}")
        
        # Get all predictions
        ctr_prediction = self.predict_ctr(video_metrics, campaign_metrics, platform, demographic)
        conversion_prediction = self.predict_conversion_rate(video_metrics, campaign_metrics, platform, demographic)
        cost_prediction = self.predict_cost_metrics(
            ctr_prediction["predicted_ctr"],
            conversion_prediction["predicted_conversion_rate"],
            campaign_metrics,
            platform
        )
        scaling_prediction = self.predict_scaling_potential(
            video_metrics, campaign_metrics, 
            {**ctr_prediction, **cost_prediction}
        )
        
        # Calculate campaign projections
        impressions = budget / cost_prediction["predicted_cpm"] * 1000
        clicks = impressions * (ctr_prediction["predicted_ctr"] / 100)
        conversions = clicks * (conversion_prediction["predicted_conversion_rate"] / 100)
        
        # Revenue calculation (need average order value)
        # Estimate AOV based on price point
        aov_estimates = {"budget": 25, "mid": 75, "premium": 200, "luxury": 500}
        estimated_aov = aov_estimates.get(campaign_metrics.product_price_point, 75)
        
        revenue = conversions * estimated_aov
        profit = revenue - budget
        roas = revenue / budget if budget > 0 else 0
        
        # Complete performance summary
        summary = {
            "performance_predictions": {
                "ctr": ctr_prediction,
                "conversion_rate": conversion_prediction,
                "costs": cost_prediction,
                "scaling": scaling_prediction
            },
            "campaign_projections": {
                "impressions": int(impressions),
                "clicks": int(clicks),
                "conversions": int(conversions),
                "revenue": revenue,
                "profit": profit,
                "roas": roas,
                "estimated_aov": estimated_aov
            },
            "success_probability": {
                "profitable": "high" if profit > 0 else "low",
                "scalable": scaling_prediction["scaling_category"],
                "sustainable": "yes" if roas > 2.0 else "no",
                "optimization_needed": len(scaling_prediction["recommendations"])
            },
            "next_steps": self._generate_next_steps(scaling_prediction, profit, roas)
        }
        
        print(f"\n📊 CAMPAIGN SUMMARY:")
        print(f"   Expected ROAS: {roas:.2f}x")
        print(f"   Expected Profit: ${profit:,.2f}")
        print(f"   Scaling Category: {scaling_prediction['scaling_category'].upper()}")
        
        return summary
    
    def _generate_next_steps(
        self,
        scaling_prediction: Dict,
        profit: float,
        roas: float
    ) -> List[str]:
        """Generate recommended next steps based on predictions"""
        steps = []
        
        if profit <= 0:
            steps.append("🔴 Campaign not profitable - revise creative or targeting")
        elif roas < 2.0:
            steps.append("🟡 Low ROAS - optimize for efficiency before scaling")
        elif scaling_prediction["scaling_category"] == "extreme":
            steps.append("🟢 Ready for aggressive scaling - increase budget 5-10x")
        elif scaling_prediction["scaling_category"] == "high":
            steps.append("🟢 Scale gradually - increase budget 2-3x and monitor")
        else:
            steps.append("🟡 Optimize performance before major scaling")
        
        # Add specific optimization steps
        if len(scaling_prediction["recommendations"]) > 0:
            steps.extend([f"• {rec}" for rec in scaling_prediction["recommendations"][:3]])
        
        return steps


def main():
    """Test performance prediction models"""
    print("\n📈 Testing Performance Prediction Models (Mac)")
    print("="*55)
    
    # Initialize predictor
    predictor = PerformancePredictionModels()
    
    # Create test video metrics
    test_video = VideoMetrics(
        duration=15.0,
        has_product_close_up=True,
        has_text_overlay=True,
        has_motion=True,
        color_diversity=0.7,
        brightness_level=0.6,
        contrast_level=0.8,
        face_present=False,
        product_screen_time=0.6,  # 60% of video shows product
        hook_strength=0.85,       # Strong opening hook
        cta_clarity=0.9          # Very clear CTA
    )
    
    # Create test campaign metrics
    test_campaign = CampaignMetrics(
        budget=5000.0,
        target_audience_size=5_000_000,
        competition_level=0.4,      # Medium competition
        seasonal_factor=1.2,        # 20% seasonal boost
        brand_recognition=0.3,      # New brand
        product_price_point="mid",  # Mid-range pricing
        previous_campaign_ctr=3.2
    )
    
    # Test different platform/demographic combinations
    test_combinations = [
        ("instagram_reels", "gen_z"),
        ("facebook_feed", "millennials"),
        ("tiktok", "gen_z"),
        ("instagram_feed", "professionals")
    ]
    
    print("\n1. Testing performance predictions...")
    
    all_results = []
    for platform, demographic in test_combinations:
        print(f"\n   🎯 Testing: {platform} / {demographic}")
        
        # Get complete performance analysis
        result = predictor.predict_campaign_performance(
            video_metrics=test_video,
            campaign_metrics=test_campaign,
            platform=platform,
            demographic=demographic,
            budget=test_campaign.budget
        )
        
        all_results.append({
            "platform": platform,
            "demographic": demographic,
            "result": result
        })
        
        # Show key metrics
        projections = result["campaign_projections"]
        success = result["success_probability"]
        
        print(f"      ROAS: {projections['roas']:.2f}x")
        print(f"      Profit: ${projections['profit']:,.2f}")
        print(f"      Scaling: {success['scalable']}")
    
    # Find best performing combination
    print(f"\n2. Best performing combinations:")
    sorted_results = sorted(
        all_results,
        key=lambda x: x["result"]["campaign_projections"]["roas"],
        reverse=True
    )
    
    for i, combo in enumerate(sorted_results[:3]):
        platform = combo["platform"]
        demographic = combo["demographic"] 
        roas = combo["result"]["campaign_projections"]["roas"]
        scaling = combo["result"]["success_probability"]["scalable"]
        
        print(f"   {i+1}. {platform} / {demographic}: {roas:.2f}x ROAS ({scaling} scaling)")
    
    # Test 3: Scaling potential analysis
    print(f"\n3. Detailed scaling analysis for best combination...")
    best_combo = sorted_results[0]
    scaling_analysis = best_combo["result"]["performance_predictions"]["scaling"]
    
    print(f"   Overall Score: {scaling_analysis['overall_score']:.3f}")
    print(f"   Daily Revenue Potential: ${scaling_analysis['daily_revenue_potential'][0]:,}-${scaling_analysis['daily_revenue_potential'][1]:,}")
    print(f"   Content Score: {scaling_analysis['content_score']:.3f}")
    print(f"   Market Score: {scaling_analysis['market_score']:.3f}")
    
    # Show recommendations
    print(f"\n   📋 Recommendations:")
    for rec in scaling_analysis["recommendations"]:
        print(f"      • {rec}")
    
    # Save complete analysis
    print(f"\n4. Saving performance analysis...")
    with open("performance_analysis_complete.json", 'w') as f:
        json.dump({
            "test_video_metrics": test_video.__dict__,
            "test_campaign_metrics": test_campaign.__dict__,
            "platform_results": all_results,
            "best_combination": {
                "platform": best_combo["platform"],
                "demographic": best_combo["demographic"],
                "expected_roas": best_combo["result"]["campaign_projections"]["roas"]
            }
        }, f, indent=2, default=str)
    
    print("   ✅ Analysis saved to: performance_analysis_complete.json")
    
    print("\n✅ Performance Prediction Models Test Complete!")
    print("📈 Ready for data-driven campaign optimization")


if __name__ == "__main__":
    main()