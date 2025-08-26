#!/usr/bin/env python3
"""
Complete Test Suite for Mac Logic Components
Validates all algorithms before RunPod deployment
NO video generation - logic validation only
"""

import unittest
import numpy as np
from PIL import Image
import json
import tempfile
import os
from pathlib import Path

# Import our modules
from corepulse_video_logic import CorePulseVideoLogic, VideoFrameInjection, FrameInjectionLevel
from product_preservation_logic import ProductPreservationLogic, PreservationMode
from product_detection_algorithms import ProductDetectionAlgorithms, DetectionMethod
from tracking_algorithms import TrackingAlgorithms, TrackingState, ProductTrack
from video_campaign_templates import VideoCampaignTemplates, CampaignType, Platform, Demographic
from performance_prediction_models import PerformancePredictionModels, VideoMetrics, CampaignMetrics


class TestCorePulseVideoLogic(unittest.TestCase):
    """Test CorePulse video control algorithms"""
    
    def setUp(self):
        self.logic = CorePulseVideoLogic()
    
    def test_frame_injection_calculation(self):
        """Test frame injection calculation"""
        injections = self.logic.calculate_frame_injections(
            video_length=10,
            template_name="product_hero_shot",
            product_regions=[],
            custom_prompts=None
        )
        
        # Should have injections for all frames (3 per frame: early, mid, late)
        self.assertEqual(len(injections), 30)  # 10 frames * 3 injections
        
        # Check injection types
        levels = [inj.level for inj in injections]
        self.assertIn(FrameInjectionLevel.TEMPORAL_EARLY, levels)
        self.assertIn(FrameInjectionLevel.SPATIAL_MID, levels)
        self.assertIn(FrameInjectionLevel.STYLE_LATE, levels)
    
    def test_injection_schedule_generation(self):
        """Test complete injection schedule generation"""
        test_config = {
            "frame_count": 5,
            "template": "product_hero_shot",
            "product_regions": []
        }
        
        schedule = self.logic.generate_injection_schedule(test_config)
        
        # Validate schedule structure
        self.assertIn("frame_injections", schedule)
        self.assertIn("consistency_rules", schedule)
        self.assertIn("metadata", schedule)
        
        # Should have injections for all frames
        self.assertEqual(len(schedule["frame_injections"]), 15)  # 5 frames * 3 injections
    
    def test_schedule_validation(self):
        """Test schedule validation logic"""
        valid_schedule = self.logic.create_test_schedule(frames=5)
        is_valid, issues = self.logic.validate_schedule(valid_schedule)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Test invalid schedule
        invalid_schedule = {"frame_injections": []}
        is_valid, issues = self.logic.validate_schedule(invalid_schedule)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)


class TestProductPreservationLogic(unittest.TestCase):
    """Test product preservation algorithms"""
    
    def setUp(self):
        self.logic = ProductPreservationLogic()
        # Create test image
        self.test_image = self._create_test_product_image()
    
    def _create_test_product_image(self) -> np.ndarray:
        """Create test image with clear product"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White background
        img[30:70, 30:70] = [255, 0, 0]  # Red square product
        return img
    
    def test_preservation_mask_calculation(self):
        """Test mask calculation for product preservation"""
        mask, metadata = self.logic.calculate_preservation_mask(self.test_image)
        
        # Should detect the red square
        self.assertGreater(metadata["coverage_percent"], 5)  # At least 5% coverage
        self.assertGreater(metadata["confidence"], 0.1)      # Some confidence
        
        # Mask should be binary
        unique_values = np.unique(mask)
        self.assertTrue(all(val in [0, 255] for val in unique_values))
    
    def test_pixel_locks_calculation(self):
        """Test pixel lock rule generation"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255  # Product region
        
        locks = self.logic.calculate_pixel_locks(
            image=self.test_image,
            mask=mask,
            lock_strength=1.0,
            tolerance_level="strict"
        )
        
        # Should generate pixel locks
        self.assertGreater(len(locks), 0)
        
        # All locks should have valid parameters
        for lock in locks:
            self.assertGreaterEqual(lock.lock_strength, 0.0)
            self.assertLessEqual(lock.lock_strength, 1.0)
            self.assertGreater(lock.priority, 0)
    
    def test_preservation_config_generation(self):
        """Test complete preservation config generation"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_pil = Image.fromarray(self.test_image)
            img_pil.save(tmp.name)
            
            config = self.logic.generate_preservation_config(tmp.name)
            
            # Clean up
            os.unlink(tmp.name)
        
        # Validate config structure
        self.assertIn("preservation_mask", config)
        self.assertIn("metadata", config)
        self.assertIn("pixel_locks", config)
        self.assertIn("deployment_info", config)


class TestProductDetectionAlgorithms(unittest.TestCase):
    """Test product detection algorithms"""
    
    def setUp(self):
        self.detector = ProductDetectionAlgorithms()
        self.test_image = self._create_test_image()
    
    def _create_test_image(self) -> np.ndarray:
        """Create test image with detectable product"""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 230  # Light gray background
        img[50:150, 50:150] = [100, 150, 200]  # Blue product
        return img
    
    def test_background_subtraction(self):
        """Test background subtraction detection"""
        result = self.detector.detect_product_comprehensive(
            self.test_image, DetectionMethod.BACKGROUND_SUBTRACTION
        )
        
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        # Should detect something
        self.assertGreater(np.sum(result.mask), 0)
    
    def test_color_clustering(self):
        """Test color clustering detection"""
        result = self.detector.detect_product_comprehensive(
            self.test_image, DetectionMethod.COLOR_CLUSTERING
        )
        
        self.assertEqual(result.method_used, DetectionMethod.COLOR_CLUSTERING)
        self.assertIsInstance(result.bbox, tuple)
        self.assertEqual(len(result.bbox), 4)
    
    def test_combined_detection(self):
        """Test combined detection method"""
        result = self.detector.detect_product_comprehensive(
            self.test_image, DetectionMethod.COMBINED
        )
        
        self.assertEqual(result.method_used, DetectionMethod.COMBINED)
        self.assertGreater(result.confidence, 0.0)
        
        # Properties should be extracted
        self.assertIsInstance(result.properties, dict)


class TestTrackingAlgorithms(unittest.TestCase):
    """Test product tracking algorithms"""
    
    def setUp(self):
        self.tracker = TrackingAlgorithms(max_disappeared=3, max_distance=50)
    
    def test_single_product_tracking(self):
        """Test tracking single product across frames"""
        # Create simple detection sequence (product moving right)
        detections = []
        for i in range(10):
            x = 50 + i * 10  # Moving right
            detections.append([{
                "bbox": (x, 100, x + 50, 150),
                "confidence": 0.8,
                "properties": {"color": "red"}
            }])
        
        tracks = self.tracker.track_products_in_sequence(detections, frame_rate=30.0)
        
        # Should create one continuous track
        self.assertGreater(len(tracks), 0)
        
        # Find longest track (should be our moving product)
        longest_track = max(tracks.values(), key=lambda t: t.age)
        self.assertGreaterEqual(longest_track.age, 8)  # Should track most frames
        self.assertEqual(longest_track.state, TrackingState.DETECTED)
    
    def test_multiple_product_tracking(self):
        """Test tracking multiple products simultaneously"""
        detections = []
        for i in range(5):
            frame_detections = [
                {"bbox": (50 + i*5, 100, 100 + i*5, 150), "confidence": 0.8, "properties": {}},  # Product 1
                {"bbox": (200, 50 + i*3, 250, 100 + i*3), "confidence": 0.7, "properties": {}}   # Product 2
            ]
            detections.append(frame_detections)
        
        tracks = self.tracker.track_products_in_sequence(detections, frame_rate=30.0)
        
        # Should track both products
        active_tracks = [t for t in tracks.values() if t.state != TrackingState.LOST]
        self.assertGreaterEqual(len(active_tracks), 2)
    
    def test_tracking_quality_metrics(self):
        """Test tracking quality calculation"""
        # Create mock tracks
        track1 = ProductTrack(
            track_id=1, current_bbox=(50, 50, 100, 100),
            current_centroid=(75, 75), velocity=(5, 0),
            confidence=0.8, state=TrackingState.DETECTED,
            age=10, missed_frames=0, history=[]
        )
        
        tracks = {1: track1}
        metrics = self.tracker.calculate_tracking_quality_metrics(tracks)
        
        self.assertIn("total_tracks", metrics)
        self.assertIn("active_tracks", metrics)
        self.assertIn("overall_quality", metrics)
        self.assertGreaterEqual(metrics["overall_quality"], 0.0)
        self.assertLessEqual(metrics["overall_quality"], 1.0)


class TestVideoCampaignTemplates(unittest.TestCase):
    """Test video campaign templates"""
    
    def setUp(self):
        self.templates = VideoCampaignTemplates()
    
    def test_template_loading(self):
        """Test template initialization"""
        # Should have all campaign types
        expected_types = [
            CampaignType.PRODUCT_HERO,
            CampaignType.LIFESTYLE_INTEGRATION,
            CampaignType.UNBOXING_REVEAL,
            CampaignType.PROBLEM_SOLUTION,
            CampaignType.SOCIAL_PROOF
        ]
        
        for campaign_type in expected_types:
            template = self.templates.get_template(campaign_type)
            self.assertIsNotNone(template)
            self.assertEqual(template.type, campaign_type)
    
    def test_platform_filtering(self):
        """Test filtering templates by platform"""
        reels_templates = self.templates.get_templates_for_platform(Platform.INSTAGRAM_REELS)
        
        self.assertGreater(len(reels_templates), 0)
        
        # All returned templates should support Instagram Reels
        for template in reels_templates:
            self.assertIn(Platform.INSTAGRAM_REELS, template.target_platforms)
    
    def test_variation_generation(self):
        """Test campaign variation generation"""
        hero_template = self.templates.get_template(CampaignType.PRODUCT_HERO)
        variations = self.templates.generate_campaign_variations(hero_template, variation_count=3)
        
        self.assertEqual(len(variations), 3)
        
        # Each variation should have required fields
        for variation in variations:
            self.assertIn("variation_id", variation)
            self.assertIn("expected_performance", variation)
            self.assertIn("modifications", variation)
    
    def test_deployment_package_creation(self):
        """Test deployment package generation"""
        template = self.templates.get_template(CampaignType.PRODUCT_HERO)
        variations = self.templates.generate_campaign_variations(template, variation_count=2)
        
        package = self.templates.create_campaign_deployment_package(
            template=template,
            variations=variations,
            target_budget=1000.0
        )
        
        # Validate package structure
        required_keys = [
            "campaign_info", "technical_specs", "variations",
            "production_requirements", "deployment_schedule", "metadata"
        ]
        
        for key in required_keys:
            self.assertIn(key, package)


class TestPerformancePredictionModels(unittest.TestCase):
    """Test performance prediction models"""
    
    def setUp(self):
        self.predictor = PerformancePredictionModels()
        self.test_video = VideoMetrics(
            duration=10.0, has_product_close_up=True, has_text_overlay=True,
            has_motion=True, color_diversity=0.5, brightness_level=0.6,
            contrast_level=0.7, face_present=False, product_screen_time=0.4,
            hook_strength=0.7, cta_clarity=0.8
        )
        self.test_campaign = CampaignMetrics(
            budget=1000.0, target_audience_size=1_000_000, competition_level=0.5,
            seasonal_factor=1.0, brand_recognition=0.5, product_price_point="mid"
        )
    
    def test_ctr_prediction(self):
        """Test CTR prediction"""
        result = self.predictor.predict_ctr(
            self.test_video, self.test_campaign, "instagram_reels", "gen_z"
        )
        
        # Should return valid CTR prediction
        self.assertIn("predicted_ctr", result)
        self.assertGreater(result["predicted_ctr"], 0.0)
        self.assertIn("confidence_min", result)
        self.assertIn("confidence_max", result)
    
    def test_conversion_prediction(self):
        """Test conversion rate prediction"""
        result = self.predictor.predict_conversion_rate(
            self.test_video, self.test_campaign, "facebook_feed", "millennials"
        )
        
        self.assertIn("predicted_conversion_rate", result)
        self.assertGreater(result["predicted_conversion_rate"], 0.0)
        self.assertIn("base_factors", result)
    
    def test_scaling_prediction(self):
        """Test scaling potential prediction"""
        # Mock performance data
        mock_performance = {"predicted_ctr": 3.0, "predicted_cpa": 25.0}
        
        result = self.predictor.predict_scaling_potential(
            self.test_video, self.test_campaign, mock_performance
        )
        
        self.assertIn("scaling_category", result)
        self.assertIn("overall_score", result)
        self.assertIn("daily_revenue_potential", result)
        self.assertIn("recommendations", result)
        
        # Score should be between 0 and 1
        self.assertGreaterEqual(result["overall_score"], 0.0)
        self.assertLessEqual(result["overall_score"], 1.0)
    
    def test_complete_performance_analysis(self):
        """Test complete performance prediction pipeline"""
        result = self.predictor.predict_campaign_performance(
            self.test_video, self.test_campaign, "tiktok", "gen_z", 1000.0
        )
        
        # Should have all sections
        required_sections = [
            "performance_predictions", "campaign_projections",
            "success_probability", "next_steps"
        ]
        
        for section in required_sections:
            self.assertIn(section, result)


class TestIntegrationLogic(unittest.TestCase):
    """Test integration between components"""
    
    def setUp(self):
        self.video_logic = CorePulseVideoLogic()
        self.preservation_logic = ProductPreservationLogic()
        self.detection_algorithms = ProductDetectionAlgorithms()
        self.tracking_algorithms = TrackingAlgorithms()
        self.templates = VideoCampaignTemplates()
        self.predictor = PerformancePredictionModels()
    
    def test_end_to_end_logic_flow(self):
        """Test complete logic flow from template to prediction"""
        # 1. Get campaign template
        template = self.templates.get_template(CampaignType.PRODUCT_HERO)
        self.assertIsNotNone(template)
        
        # 2. Generate injection schedule
        video_config = {
            "frame_count": 10,
            "template": "product_hero_shot",
            "product_regions": []
        }
        schedule = self.video_logic.generate_injection_schedule(video_config)
        self.assertIn("frame_injections", schedule)
        
        # 3. Create test detection sequence
        detections = [[{"bbox": (50, 50, 100, 100), "confidence": 0.8, "properties": {}}] for _ in range(10)]
        tracks = self.tracking_algorithms.track_products_in_sequence(detections, 30.0)
        self.assertGreater(len(tracks), 0)
        
        # 4. Generate performance prediction
        test_video = VideoMetrics(
            duration=10.0, has_product_close_up=True, has_text_overlay=True,
            has_motion=True, color_diversity=0.5, brightness_level=0.6,
            contrast_level=0.7, face_present=False, product_screen_time=0.5,
            hook_strength=0.8, cta_clarity=0.9
        )
        test_campaign = CampaignMetrics(
            budget=1000.0, target_audience_size=1_000_000, competition_level=0.3,
            seasonal_factor=1.0, brand_recognition=0.4, product_price_point="mid"
        )
        
        performance = self.predictor.predict_campaign_performance(
            test_video, test_campaign, "instagram_reels", "gen_z", 1000.0
        )
        
        self.assertIn("success_probability", performance)
        
        print("   ✅ End-to-end logic flow successful")
    
    def test_data_flow_compatibility(self):
        """Test that data structures are compatible between components"""
        # Test that injection schedule can be consumed by tracking
        schedule = self.video_logic.create_test_schedule()
        
        # Validate schedule format
        is_valid, issues = self.video_logic.validate_schedule(schedule)
        self.assertTrue(is_valid, f"Schedule validation failed: {issues}")
        
        # Test that preservation config has required fields
        test_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(test_img).save(tmp.name)
            config = self.preservation_logic.generate_preservation_config(tmp.name)
            os.unlink(tmp.name)
        
        required_fields = ["preservation_mask", "pixel_locks", "deployment_info"]
        for field in required_fields:
            self.assertIn(field, config)
        
        print("   ✅ Data flow compatibility verified")


def run_comprehensive_test_suite():
    """Run complete test suite with reporting"""
    print("\n🧪 Running Comprehensive Mac Logic Test Suite")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestCorePulseVideoLogic,
        TestProductPreservationLogic,
        TestProductDetectionAlgorithms,
        TestTrackingAlgorithms,
        TestVideoCampaignTemplates,
        TestPerformancePredictionModels,
        TestIntegrationLogic
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Generate summary
    print(f"\n{'='*60}")
    print(f"TEST SUITE SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n🔥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Determine readiness
    if len(result.failures) == 0 and len(result.errors) == 0:
        print(f"\n✅ ALL TESTS PASSED - READY FOR RUNPOD DEPLOYMENT")
        readiness = "ready"
    elif len(result.failures) + len(result.errors) <= 2:
        print(f"\n🟡 MOSTLY READY - MINOR ISSUES TO RESOLVE")
        readiness = "mostly_ready"
    else:
        print(f"\n🔴 SIGNIFICANT ISSUES - NEED FIXES BEFORE DEPLOYMENT")
        readiness = "needs_work"
    
    # Save test report
    test_report = {
        "timestamp": "test_run",
        "total_tests": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun,
        "readiness_status": readiness,
        "failure_details": [str(f[0]) for f in result.failures],
        "error_details": [str(e[0]) for e in result.errors]
    }
    
    with open("mac_logic_test_report.json", 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n📊 Test report saved to: mac_logic_test_report.json")
    
    return readiness == "ready"


if __name__ == "__main__":
    success = run_comprehensive_test_suite()