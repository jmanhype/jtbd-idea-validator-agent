#!/usr/bin/env python3
"""
Product Detection Algorithms
Advanced product detection and segmentation logic
OpenCV operations only - NO ML inference (Mac development)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum
import cv2
from PIL import Image, ImageFilter
import json
import hashlib
from pathlib import Path
import colorsys

class DetectionMethod(Enum):
    """Different detection approaches"""
    BACKGROUND_SUBTRACTION = "background_subtraction"
    COLOR_CLUSTERING = "color_clustering"
    EDGE_DETECTION = "edge_detection"
    CONTOUR_ANALYSIS = "contour_analysis"
    TEMPLATE_MATCHING = "template_matching"
    COMBINED = "combined"


@dataclass
class DetectionResult:
    """Result of product detection"""
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    method_used: DetectionMethod
    properties: Dict  # Color, shape, size properties
    contours: List[np.ndarray] = None


class BoundingBox(NamedTuple):
    """Structured bounding box"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class ProductDetectionAlgorithms:
    """
    Advanced algorithms for product detection and segmentation
    All OpenCV and math operations - no ML inference
    """
    
    def __init__(self):
        self.detection_cache = {}
        self.color_profiles = {}
        
        # Detection parameters
        self.params = {
            "bg_threshold": 0.15,          # Background similarity threshold
            "edge_low": 50,                # Canny low threshold
            "edge_high": 150,              # Canny high threshold
            "min_contour_area": 500,       # Minimum contour area
            "max_contour_area": 50000,     # Maximum contour area
            "color_clusters": 3,           # K-means clusters
            "morphology_kernel": 5,        # Morphology kernel size
            "gaussian_blur": 3             # Gaussian blur radius
        }
    
    def detect_product_comprehensive(
        self,
        image: np.ndarray,
        method: DetectionMethod = DetectionMethod.COMBINED,
        reference_image: Optional[np.ndarray] = None
    ) -> DetectionResult:
        """
        Comprehensive product detection using multiple methods
        """
        print(f"🔍 Detecting product using: {method.value}")
        print(f"   Image shape: {image.shape}")
        
        if method == DetectionMethod.COMBINED:
            return self._combined_detection(image, reference_image)
        elif method == DetectionMethod.BACKGROUND_SUBTRACTION:
            return self._background_subtraction_detection(image)
        elif method == DetectionMethod.COLOR_CLUSTERING:
            return self._color_clustering_detection(image)
        elif method == DetectionMethod.EDGE_DETECTION:
            return self._edge_detection(image)
        elif method == DetectionMethod.CONTOUR_ANALYSIS:
            return self._contour_analysis_detection(image)
        elif method == DetectionMethod.TEMPLATE_MATCHING:
            return self._template_matching_detection(image, reference_image)
        else:
            return self._combined_detection(image, reference_image)
    
    def _combined_detection(
        self,
        image: np.ndarray,
        reference_image: Optional[np.ndarray] = None
    ) -> DetectionResult:
        """
        Combine multiple detection methods for best results
        """
        print("   🧠 Running combined detection...")
        
        # Run all detection methods
        methods = [
            self._background_subtraction_detection,
            self._color_clustering_detection,
            self._edge_detection,
            self._contour_analysis_detection
        ]
        
        results = []
        for method in methods:
            try:
                result = method(image)
                results.append(result)
            except Exception as e:
                print(f"   ⚠️ Method failed: {e}")
                continue
        
        if not results:
            # Fallback: return full image
            return DetectionResult(
                mask=np.ones(image.shape[:2], dtype=np.uint8) * 255,
                bbox=(0, 0, image.shape[1], image.shape[0]),
                confidence=0.1,
                method_used=DetectionMethod.COMBINED,
                properties={}
            )
        
        # Combine masks using weighted voting
        combined_mask = self._combine_detection_masks([r.mask for r in results])
        
        # Calculate combined confidence
        avg_confidence = np.mean([r.confidence for r in results])
        
        # Find best bounding box
        bbox = self._calculate_optimal_bbox(combined_mask)
        
        # Extract properties
        properties = self._extract_product_properties(image, combined_mask)
        
        # Find contours
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        result = DetectionResult(
            mask=combined_mask,
            bbox=bbox,
            confidence=avg_confidence,
            method_used=DetectionMethod.COMBINED,
            properties=properties,
            contours=contours
        )
        
        print(f"   ✅ Combined detection: {result.confidence:.3f} confidence")
        return result
    
    def _background_subtraction_detection(self, image: np.ndarray) -> DetectionResult:
        """
        Detect product by subtracting estimated background
        """
        print("     📷 Background subtraction...")
        
        if len(image.shape) != 3:
            # Grayscale fallback
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        else:
            # Estimate background from corners and edges
            bg_estimate = self._estimate_background_color(image)
            
            # Calculate color difference
            diff = np.sqrt(np.sum((image - bg_estimate) ** 2, axis=2))
            max_diff = np.sqrt(3 * 255 ** 2)
            
            # Threshold
            threshold = self.params["bg_threshold"] * max_diff
            mask = (diff > threshold).astype(np.uint8) * 255
            
            # Clean up mask
            mask = self._clean_mask(mask)
        
        bbox = self._calculate_optimal_bbox(mask)
        confidence = self._calculate_mask_confidence(mask, image.shape[:2])
        
        return DetectionResult(
            mask=mask,
            bbox=bbox,
            confidence=confidence,
            method_used=DetectionMethod.BACKGROUND_SUBTRACTION,
            properties={"background_threshold": self.params["bg_threshold"]}
        )
    
    def _color_clustering_detection(self, image: np.ndarray) -> DetectionResult:
        """
        Detect product using color clustering (K-means)
        """
        print("     🎨 Color clustering...")
        
        if len(image.shape) != 3:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        else:
            # Reshape for clustering
            pixels = image.reshape(-1, 3).astype(np.float32)
            
            # K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(
                pixels, self.params["color_clusters"], None, 
                criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Reshape labels back
            labels = labels.reshape(image.shape[:2])
            
            # Find product cluster (not background)
            product_cluster = self._identify_product_cluster(labels, centers)
            
            # Create mask
            mask = (labels == product_cluster).astype(np.uint8) * 255
            
            # Clean up
            mask = self._clean_mask(mask)
        
        bbox = self._calculate_optimal_bbox(mask)
        confidence = self._calculate_mask_confidence(mask, image.shape[:2])
        
        return DetectionResult(
            mask=mask,
            bbox=bbox,
            confidence=confidence,
            method_used=DetectionMethod.COLOR_CLUSTERING,
            properties={"clusters": self.params["color_clusters"]}
        )
    
    def _edge_detection(self, image: np.ndarray) -> DetectionResult:
        """
        Detect product using edge detection and contour analysis
        """
        print("     ✏️ Edge detection...")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.params["gaussian_blur"], self.params["gaussian_blur"]), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.params["edge_low"], self.params["edge_high"])
        
        # Morphological operations to close gaps
        kernel = np.ones((self.params["morphology_kernel"], self.params["morphology_kernel"]), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Fill enclosed regions
        mask = self._fill_enclosed_regions(closed)
        
        bbox = self._calculate_optimal_bbox(mask)
        confidence = self._calculate_mask_confidence(mask, image.shape[:2])
        
        return DetectionResult(
            mask=mask,
            bbox=bbox,
            confidence=confidence,
            method_used=DetectionMethod.EDGE_DETECTION,
            properties={
                "edge_low": self.params["edge_low"],
                "edge_high": self.params["edge_high"]
            }
        )
    
    def _contour_analysis_detection(self, image: np.ndarray) -> DetectionResult:
        """
        Detect product using contour analysis
        """
        print("     🔗 Contour analysis...")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [
            c for c in contours 
            if self.params["min_contour_area"] <= cv2.contourArea(c) <= self.params["max_contour_area"]
        ]
        
        if not valid_contours:
            # No valid contours found
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            # Use largest valid contour
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # Create mask from contour
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        bbox = self._calculate_optimal_bbox(mask)
        confidence = self._calculate_mask_confidence(mask, image.shape[:2])
        
        return DetectionResult(
            mask=mask,
            bbox=bbox,
            confidence=confidence,
            method_used=DetectionMethod.CONTOUR_ANALYSIS,
            properties={"contours_found": len(valid_contours)},
            contours=valid_contours
        )
    
    def _template_matching_detection(
        self,
        image: np.ndarray,
        template: Optional[np.ndarray] = None
    ) -> DetectionResult:
        """
        Detect product using template matching (if reference provided)
        """
        print("     🎯 Template matching...")
        
        if template is None:
            # No template provided - fallback
            return self._background_subtraction_detection(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        else:
            template_gray = template.copy()
        
        # Template matching
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find best match
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Create mask from template match
        h, w = template_gray.shape
        x, y = max_loc
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        
        bbox = (x, y, x + w, y + h)
        confidence = float(max_val)  # Template matching confidence
        
        return DetectionResult(
            mask=mask,
            bbox=bbox,
            confidence=confidence,
            method_used=DetectionMethod.TEMPLATE_MATCHING,
            properties={"match_score": max_val}
        )
    
    def _estimate_background_color(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate background color from image corners and edges
        """
        h, w = image.shape[:2]
        corner_size = min(20, h // 8, w // 8)
        
        # Sample corners and edges
        samples = []
        
        # Corners
        corners = [
            image[:corner_size, :corner_size],           # Top-left
            image[:corner_size, -corner_size:],          # Top-right
            image[-corner_size:, :corner_size],          # Bottom-left
            image[-corner_size:, -corner_size:]          # Bottom-right
        ]
        
        # Edges (excluding corners)
        edge_size = max(5, min(h, w) // 20)
        edges = [
            image[:edge_size, corner_size:-corner_size],     # Top edge
            image[-edge_size:, corner_size:-corner_size],    # Bottom edge
            image[corner_size:-corner_size, :edge_size],     # Left edge
            image[corner_size:-corner_size, -edge_size:]     # Right edge
        ]
        
        # Collect all samples
        all_samples = corners + edges
        for sample in all_samples:
            if sample.size > 0:
                samples.append(np.mean(sample.reshape(-1, 3), axis=0))
        
        if samples:
            return np.mean(samples, axis=0)
        else:
            return np.array([128, 128, 128])  # Gray fallback
    
    def _identify_product_cluster(
        self,
        labels: np.ndarray,
        centers: np.ndarray
    ) -> int:
        """
        Identify which cluster represents the product (not background)
        """
        # Count pixels in corners for each cluster
        h, w = labels.shape
        corner_size = min(10, h // 10, w // 10)
        
        corner_regions = [
            labels[:corner_size, :corner_size],
            labels[:corner_size, -corner_size:],
            labels[-corner_size:, :corner_size],
            labels[-corner_size:, -corner_size:]
        ]
        
        # Count cluster occurrences in corners
        cluster_counts = {}
        for region in corner_regions:
            for cluster_id in np.unique(region):
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + np.sum(region == cluster_id)
        
        # Background cluster is most common in corners
        if cluster_counts:
            bg_cluster = max(cluster_counts.keys(), key=lambda k: cluster_counts[k])
            # Product cluster is the one with most distinct color from background
            other_clusters = [i for i in range(len(centers)) if i != bg_cluster]
            if other_clusters:
                # Choose cluster with maximum color distance from background
                bg_color = centers[bg_cluster]
                distances = [np.linalg.norm(centers[i] - bg_color) for i in other_clusters]
                product_cluster = other_clusters[np.argmax(distances)]
            else:
                product_cluster = bg_cluster  # Fallback
        else:
            product_cluster = 0  # Fallback
        
        return product_cluster
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up mask using morphological operations
        """
        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small holes
        kernel_medium = np.ones((5, 5), np.uint8)
        filled = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # Smooth edges
        filled = cv2.GaussianBlur(filled, (3, 3), 0)
        _, filled = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY)
        
        return filled
    
    def _fill_enclosed_regions(self, edges: np.ndarray) -> np.ndarray:
        """
        Fill enclosed regions from edge map
        """
        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        mask = np.zeros_like(edges)
        
        # Fill valid contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.params["min_contour_area"] <= area <= self.params["max_contour_area"]:
                cv2.fillPoly(mask, [contour], 255)
        
        return mask
    
    def _combine_detection_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        Combine multiple detection masks using weighted voting
        """
        if not masks:
            return np.zeros((100, 100), dtype=np.uint8)
        
        # Ensure all masks are same size
        target_shape = masks[0].shape
        normalized_masks = []
        
        for mask in masks:
            if mask.shape != target_shape:
                mask = cv2.resize(mask, (target_shape[1], target_shape[0]))
            normalized_masks.append(mask.astype(np.float32) / 255.0)
        
        # Weighted combination
        weights = [1.0, 0.8, 0.6, 0.4]  # Decreasing weights
        weights = weights[:len(normalized_masks)]
        
        combined = np.zeros_like(normalized_masks[0])
        total_weight = 0
        
        for mask, weight in zip(normalized_masks, weights):
            combined += mask * weight
            total_weight += weight
        
        combined /= total_weight
        
        # Threshold to binary
        binary_combined = (combined > 0.4).astype(np.uint8) * 255
        
        return self._clean_mask(binary_combined)
    
    def _calculate_optimal_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate optimal bounding box from mask
        """
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return (0, 0, mask.shape[1], mask.shape[0])
        
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # Add small padding
        padding = 2
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(mask.shape[0], y_max + padding)
        x_max = min(mask.shape[1], x_max + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def _calculate_mask_confidence(
        self,
        mask: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> float:
        """
        Calculate confidence score for detection mask
        """
        if mask.size == 0:
            return 0.0
        
        # Coverage factor
        total_pixels = image_shape[0] * image_shape[1]
        mask_pixels = np.sum(mask > 0)
        coverage = mask_pixels / total_pixels
        
        # Coverage should be reasonable (not too small, not too large)
        if coverage < 0.01:  # Less than 1%
            coverage_score = coverage * 10  # Penalty for too small
        elif coverage > 0.8:  # More than 80%
            coverage_score = (1.0 - coverage) * 5  # Penalty for too large
        else:
            coverage_score = min(1.0, coverage * 3)  # Good range
        
        # Compactness factor (how well filled is the bounding box)
        bbox = self._calculate_optimal_bbox(mask)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if bbox_area > 0:
            compactness = mask_pixels / bbox_area
        else:
            compactness = 0.0
        
        # Shape regularity (not too fragmented)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest_contour)
            if hull_area > 0:
                solidity = contour_area / hull_area
            else:
                solidity = 0.0
        else:
            solidity = 0.0
        
        # Final confidence
        confidence = (
            coverage_score * 0.4 +      # 40% from coverage
            compactness * 0.3 +         # 30% from compactness
            solidity * 0.3              # 30% from shape regularity
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_product_properties(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Dict:
        """
        Extract properties of detected product
        """
        properties = {}
        
        # Get product pixels
        product_pixels = image[mask > 0]
        
        if len(product_pixels) > 0:
            # Color properties
            if len(image.shape) == 3:
                mean_color = np.mean(product_pixels, axis=0)
                properties["mean_color"] = mean_color.tolist()
                properties["dominant_color"] = self._get_dominant_color(product_pixels)
                
                # Color diversity
                color_std = np.std(product_pixels, axis=0)
                properties["color_diversity"] = np.mean(color_std)
            
            # Shape properties
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Area and perimeter
                properties["area"] = int(cv2.contourArea(largest_contour))
                properties["perimeter"] = int(cv2.arcLength(largest_contour, True))
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                properties["aspect_ratio"] = w / max(1, h)
                
                # Circularity
                if properties["perimeter"] > 0:
                    circularity = 4 * np.pi * properties["area"] / (properties["perimeter"] ** 2)
                    properties["circularity"] = circularity
                
                # Convex hull properties
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    properties["solidity"] = properties["area"] / hull_area
        
        return properties
    
    def _get_dominant_color(self, pixels: np.ndarray) -> List[float]:
        """
        Get dominant color using K-means clustering
        """
        if len(pixels) == 0:
            return [0, 0, 0]
        
        # K-means to find dominant color
        pixels_float = pixels.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels_float, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        dominant = centers[0]
        return dominant.tolist()


def main():
    """Test product detection algorithms"""
    print("\n🔍 Testing Product Detection Algorithms (Mac)")
    print("="*50)
    
    # Initialize detector
    detector = ProductDetectionAlgorithms()
    
    # Test with existing product image
    test_image_path = "test_product_watch.png"
    
    if Path(test_image_path).exists():
        print(f"\n1. Testing with real product: {test_image_path}")
        
        # Load image
        pil_image = Image.open(test_image_path)
        if pil_image.mode == 'RGBA':
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            pil_image = Image.alpha_composite(background.convert('RGBA'), pil_image)
        image = np.array(pil_image.convert('RGB'))
        
        # Test different detection methods
        methods = [
            DetectionMethod.BACKGROUND_SUBTRACTION,
            DetectionMethod.COLOR_CLUSTERING,
            DetectionMethod.EDGE_DETECTION,
            DetectionMethod.CONTOUR_ANALYSIS,
            DetectionMethod.COMBINED
        ]
        
        results = []
        for method in methods:
            result = detector.detect_product_comprehensive(image, method)
            results.append((method.value, result))
            
            print(f"   {method.value}: {result.confidence:.3f} confidence")
            print(f"     BBox: {result.bbox}")
            if result.properties:
                print(f"     Properties: {len(result.properties)} attributes")
        
        # Find best result
        best_method, best_result = max(results, key=lambda x: x[1].confidence)
        print(f"\n   🏆 Best method: {best_method} ({best_result.confidence:.3f})")
        
        # Save best mask
        mask_image = Image.fromarray(best_result.mask)
        mask_image.save("detected_product_mask.png")
        print(f"   💾 Best mask saved to: detected_product_mask.png")
    
    else:
        print("\n1. Creating test image for detection...")
        
        # Create test image with multiple objects
        test_img = np.ones((300, 300, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add red circle (main product)
        cv2.circle(test_img, (100, 100), 40, (255, 50, 50), -1)
        
        # Add blue rectangle (secondary object)
        cv2.rectangle(test_img, (180, 80), (250, 150), (50, 50, 255), -1)
        
        # Add some noise
        noise = np.random.randint(0, 30, test_img.shape, dtype=np.uint8)
        test_img = np.clip(test_img.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Save test image
        test_pil = Image.fromarray(test_img)
        test_pil.save("test_detection_image.png")
        print("   💾 Test image saved to: test_detection_image.png")
        
        # Test detection
        result = detector.detect_product_comprehensive(test_img, DetectionMethod.COMBINED)
        print(f"   🔍 Detection confidence: {result.confidence:.3f}")
        print(f"   📦 Bounding box: {result.bbox}")
        
        # Save result mask
        mask_image = Image.fromarray(result.mask)
        mask_image.save("test_detection_mask.png")
        print("   💾 Detection mask saved to: test_detection_mask.png")
    
    print("\n2. Testing parameter adjustment...")
    
    # Test with different parameters
    original_params = detector.params.copy()
    
    # More sensitive detection
    detector.params["bg_threshold"] = 0.08
    detector.params["min_contour_area"] = 200
    
    if Path(test_image_path).exists():
        result_sensitive = detector.detect_product_comprehensive(
            image, DetectionMethod.COMBINED
        )
        print(f"   Sensitive params: {result_sensitive.confidence:.3f} confidence")
    
    # Restore original parameters
    detector.params = original_params
    
    print("\n✅ Product Detection Algorithms Test Complete!")
    print("🔍 Ready for video frame analysis")


if __name__ == "__main__":
    main()