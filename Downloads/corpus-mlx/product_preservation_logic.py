#!/usr/bin/env python3
"""
Product Preservation Logic
Zero-hallucination algorithms for product locking
Math and logic only - NO actual video generation (Mac development)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import cv2
from PIL import Image
import json
import hashlib
from pathlib import Path

class PreservationMode(Enum):
    """Different preservation strategies"""
    ABSOLUTE = "absolute"        # 100% pixel preservation
    SOFT_EDGE = "soft_edge"     # Preserve center, blend edges
    TRACKED = "tracked"         # Follow movement while preserving
    CONTEXTUAL = "contextual"   # Preserve while adapting to context


@dataclass
class PreservationRegion:
    """Region to preserve with specific parameters"""
    mask: np.ndarray           # Binary mask of region
    mode: PreservationMode     # How to preserve
    confidence: float          # Confidence in region detection
    edge_feather: int = 0      # Pixels to feather at edges
    temporal_stability: float = 1.0  # How stable across frames


@dataclass
class PixelLockRule:
    """Rules for locking specific pixels"""
    pixel_indices: np.ndarray  # (N, 2) array of pixel coordinates
    lock_strength: float       # 0.0 = no lock, 1.0 = absolute lock
    tolerance: float          # Allowed color variation (0.0 = exact, 1.0 = any)
    priority: int             # Lock priority (higher = more important)


class ProductPreservationLogic:
    """
    Algorithms for zero-hallucination product preservation
    All math and logic - no actual image generation
    """
    
    def __init__(self):
        self.preservation_regions = []
        self.pixel_locks = []
        self.reference_data = {}
        
        # Color space tolerances for preservation
        self.color_tolerances = {
            "exact": 0.01,      # 1% RGB variation
            "strict": 0.03,     # 3% RGB variation  
            "moderate": 0.05,   # 5% RGB variation
            "loose": 0.10       # 10% RGB variation
        }
    
    def calculate_preservation_mask(
        self,
        product_image: np.ndarray,
        background_removal_threshold: float = 0.1,
        edge_refinement: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Calculate precise mask for product preservation
        Returns mask and metadata about the region
        """
        print(f"🔍 Calculating preservation mask...")
        print(f"   Image shape: {product_image.shape}")
        print(f"   BG threshold: {background_removal_threshold}")
        
        # Convert to different color spaces for analysis
        if len(product_image.shape) == 3:
            gray = cv2.cvtColor(product_image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(product_image, cv2.COLOR_RGB2HSV)
        else:
            gray = product_image
            hsv = None
        
        # Method 1: Background subtraction
        bg_mask = self._detect_background_mask(product_image, background_removal_threshold)
        
        # Method 2: Edge-based detection
        edge_mask = self._detect_edge_mask(gray)
        
        # Method 3: Color clustering
        color_mask = self._detect_color_mask(product_image, hsv)
        
        # Combine methods
        combined_mask = self._combine_masks([bg_mask, edge_mask, color_mask])
        
        # Refine edges if requested
        if edge_refinement:
            combined_mask = self._refine_mask_edges(combined_mask)
        
        # Calculate metadata
        metadata = self._analyze_preservation_region(product_image, combined_mask)
        
        print(f"   ✅ Mask coverage: {metadata['coverage_percent']:.1f}%")
        print(f"   ✅ Region confidence: {metadata['confidence']:.3f}")
        
        return combined_mask, metadata
    
    def _detect_background_mask(
        self,
        image: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Detect background using color similarity"""
        if len(image.shape) != 3:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        # Assume corners are background
        corner_size = min(20, image.shape[0] // 10, image.shape[1] // 10)
        corners = [
            image[:corner_size, :corner_size],           # Top-left
            image[:corner_size, -corner_size:],          # Top-right  
            image[-corner_size:, :corner_size],          # Bottom-left
            image[-corner_size:, -corner_size:]          # Bottom-right
        ]
        
        # Average corner colors (likely background)
        bg_colors = [np.mean(corner.reshape(-1, 3), axis=0) for corner in corners]
        avg_bg_color = np.mean(bg_colors, axis=0)
        
        # Calculate distance from background color
        color_diff = np.sqrt(np.sum((image - avg_bg_color) ** 2, axis=2))
        max_diff = np.sqrt(3 * 255 ** 2)  # Maximum possible color distance
        normalized_diff = color_diff / max_diff
        
        # Create mask (1 = foreground/product, 0 = background)
        mask = (normalized_diff > threshold).astype(np.uint8) * 255
        
        return mask
    
    def _detect_edge_mask(self, gray_image: np.ndarray) -> np.ndarray:
        """Detect product edges using Canny edge detection"""
        # Apply Gaussian blur first
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to create regions
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Fill enclosed regions
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray_image)
        if contours:
            # Find largest contour (likely the product)
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        return mask
    
    def _detect_color_mask(
        self,
        image: np.ndarray,
        hsv: Optional[np.ndarray]
    ) -> np.ndarray:
        """Detect product using color clustering"""
        if len(image.shape) != 3:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        # Reshape for clustering
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering (assume 2 clusters: background + product)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Reshape labels back to image shape
        labels = labels.reshape(image.shape[:2])
        
        # Determine which cluster is the product (not corners)
        corner_labels = [
            labels[0, 0], labels[0, -1],
            labels[-1, 0], labels[-1, -1]
        ]
        
        # Most common corner label is likely background
        bg_label = max(set(corner_labels), key=corner_labels.count)
        product_label = 1 - bg_label  # The other cluster
        
        # Create mask
        mask = (labels == product_label).astype(np.uint8) * 255
        
        return mask
    
    def _combine_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """Combine multiple masks using weighted voting"""
        if not masks:
            return np.zeros((100, 100), dtype=np.uint8)
        
        # Ensure all masks are same size
        target_shape = masks[0].shape
        normalized_masks = []
        
        for mask in masks:
            if mask.shape != target_shape:
                mask = cv2.resize(mask, (target_shape[1], target_shape[0]))
            
            # Normalize to 0-1 range
            normalized = mask.astype(np.float32) / 255.0
            normalized_masks.append(normalized)
        
        # Weighted average (equal weights for now)
        weights = [1.0, 0.8, 0.6]  # Background, Edge, Color respectively
        weights = weights[:len(normalized_masks)]
        
        combined = np.zeros_like(normalized_masks[0])
        total_weight = 0
        
        for mask, weight in zip(normalized_masks, weights):
            combined += mask * weight
            total_weight += weight
        
        combined /= total_weight
        
        # Threshold to binary mask
        binary_mask = (combined > 0.5).astype(np.uint8) * 255
        
        return binary_mask
    
    def _refine_mask_edges(self, mask: np.ndarray) -> np.ndarray:
        """Refine mask edges using morphological operations"""
        # Remove noise
        kernel_small = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Close gaps
        kernel_medium = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # Smooth edges
        kernel_large = np.ones((7, 7), np.uint8)
        smoothed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_large)
        
        return smoothed
    
    def _analyze_preservation_region(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Dict:
        """Analyze the preservation region for metadata"""
        total_pixels = mask.size
        preserved_pixels = np.sum(mask > 0)
        coverage_percent = (preserved_pixels / total_pixels) * 100
        
        # Calculate bounding box
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:
            bbox = (
                np.min(coords[1]),  # x_min
                np.min(coords[0]),  # y_min
                np.max(coords[1]),  # x_max
                np.max(coords[0])   # y_max
            )
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            fill_ratio = preserved_pixels / max(1, bbox_area)
        else:
            bbox = (0, 0, 0, 0)
            fill_ratio = 0.0
        
        # Calculate color statistics in preserved region
        if len(image.shape) == 3:
            preserved_colors = image[mask > 0]
            if len(preserved_colors) > 0:
                color_mean = np.mean(preserved_colors, axis=0)
                color_std = np.std(preserved_colors, axis=0)
                color_diversity = np.mean(color_std)
            else:
                color_mean = np.array([0, 0, 0])
                color_diversity = 0.0
        else:
            color_mean = np.array([0])
            color_diversity = 0.0
        
        # Calculate confidence score
        confidence = min(1.0, (
            (coverage_percent / 100) * 0.4 +      # Coverage contribution
            fill_ratio * 0.3 +                    # Shape quality
            min(color_diversity / 50, 1.0) * 0.3  # Color diversity
        ))
        
        return {
            "coverage_percent": coverage_percent,
            "preserved_pixels": preserved_pixels,
            "bbox": bbox,
            "fill_ratio": fill_ratio,
            "color_mean": color_mean.tolist(),
            "color_diversity": float(color_diversity),
            "confidence": confidence
        }
    
    def calculate_pixel_locks(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        lock_strength: float = 1.0,
        tolerance_level: str = "strict"
    ) -> List[PixelLockRule]:
        """
        Calculate specific pixel locks for absolute preservation
        Returns list of pixel lock rules
        """
        print(f"🔒 Calculating pixel locks...")
        print(f"   Lock strength: {lock_strength}")
        print(f"   Tolerance: {tolerance_level}")
        
        # Get pixel coordinates to lock
        lock_coords = np.where(mask > 0)
        lock_pixels = np.column_stack((lock_coords[1], lock_coords[0]))  # (x, y) format
        
        if len(lock_pixels) == 0:
            print("   ⚠️ No pixels to lock!")
            return []
        
        # Get tolerance value
        tolerance = self.color_tolerances.get(tolerance_level, 0.03)
        
        # Create different priority zones
        rules = []
        
        # High priority: Center of product (most important)
        center_mask = self._create_center_mask(mask, factor=0.3)
        center_coords = np.where(center_mask > 0)
        if len(center_coords[0]) > 0:
            center_pixels = np.column_stack((center_coords[1], center_coords[0]))
            rules.append(PixelLockRule(
                pixel_indices=center_pixels,
                lock_strength=lock_strength,
                tolerance=tolerance * 0.5,  # Stricter tolerance for center
                priority=10
            ))
        
        # Medium priority: Edges and details  
        edge_mask = self._create_edge_mask(mask)
        edge_coords = np.where(edge_mask > 0)
        if len(edge_coords[0]) > 0:
            edge_pixels = np.column_stack((edge_coords[1], edge_coords[0]))
            rules.append(PixelLockRule(
                pixel_indices=edge_pixels,
                lock_strength=lock_strength * 0.9,
                tolerance=tolerance,
                priority=8
            ))
        
        # Lower priority: Full region
        rules.append(PixelLockRule(
            pixel_indices=lock_pixels,
            lock_strength=lock_strength * 0.8,
            tolerance=tolerance * 1.2,  # More forgiving for full region
            priority=5
        ))
        
        total_locked_pixels = sum(len(rule.pixel_indices) for rule in rules)
        print(f"   ✅ Created {len(rules)} lock rules")
        print(f"   ✅ Total locked pixels: {total_locked_pixels}")
        
        return rules
    
    def _create_center_mask(self, mask: np.ndarray, factor: float = 0.3) -> np.ndarray:
        """Create mask for center region of product"""
        # Find bounding box
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return np.zeros_like(mask)
        
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # Calculate center region
        height, width = y_max - y_min, x_max - x_min
        center_h, center_w = int(height * factor), int(width * factor)
        
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        
        # Create center mask
        center_mask = np.zeros_like(mask)
        y1 = max(0, center_y - center_h // 2)
        y2 = min(mask.shape[0], center_y + center_h // 2)
        x1 = max(0, center_x - center_w // 2)
        x2 = min(mask.shape[1], center_x + center_w // 2)
        
        center_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        
        return center_mask
    
    def _create_edge_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create mask for edge regions"""
        # Morphological operations to find edges
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        edges = mask - eroded
        
        return edges
    
    def calculate_preservation_strength(
        self,
        frame_index: int,
        total_frames: int,
        movement_speed: float = 0.0
    ) -> float:
        """
        Calculate preservation strength based on temporal factors
        Higher strength when movement is slow or at key frames
        """
        # Base strength
        base_strength = 1.0
        
        # Reduce strength if moving fast (allow slight adaptation)
        movement_factor = max(0.8, 1.0 - movement_speed * 0.3)
        
        # Increase strength at key frames (start, middle, end)
        frame_position = frame_index / max(1, total_frames - 1)
        key_positions = [0.0, 0.5, 1.0]  # Start, middle, end
        key_factor = 1.0
        
        for key_pos in key_positions:
            distance = abs(frame_position - key_pos)
            if distance < 0.1:  # Within 10% of key frame
                key_factor = 1.2
                break
        
        final_strength = base_strength * movement_factor * key_factor
        return min(1.0, final_strength)
    
    def generate_preservation_config(
        self,
        product_image_path: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate complete preservation configuration
        This gets sent to RunPod for execution
        """
        print(f"\n🛡️ Generating preservation config for: {product_image_path}")
        
        # Load image and convert to RGB if needed
        try:
            pil_image = Image.open(product_image_path)
            if pil_image.mode == 'RGBA':
                # Convert RGBA to RGB with white background
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background.convert('RGBA'), pil_image)
            image = np.array(pil_image.convert('RGB'))
        except Exception as e:
            print(f"   ❌ Error loading image: {e}")
            return {}
        
        # Calculate preservation mask
        mask, metadata = self.calculate_preservation_mask(image)
        
        # Calculate pixel locks
        pixel_locks = self.calculate_pixel_locks(
            image=image,
            mask=mask,
            lock_strength=1.0,
            tolerance_level="strict"
        )
        
        # Create configuration (convert numpy types for JSON serialization)
        config = {
            "version": "1.0",
            "source_image": product_image_path,
            "preservation_mask": mask.tolist(),
            "metadata": {
                "coverage_percent": float(metadata["coverage_percent"]),
                "preserved_pixels": int(metadata["preserved_pixels"]),
                "bbox": [int(x) for x in metadata["bbox"]],
                "fill_ratio": float(metadata["fill_ratio"]),
                "color_mean": [float(x) for x in metadata["color_mean"]],
                "color_diversity": float(metadata["color_diversity"]),
                "confidence": float(metadata["confidence"])
            },
            "pixel_locks": [
                {
                    "pixel_indices": rule.pixel_indices.tolist(),
                    "lock_strength": float(rule.lock_strength),
                    "tolerance": float(rule.tolerance),
                    "priority": int(rule.priority)
                }
                for rule in pixel_locks
            ],
            "preservation_settings": {
                "mode": "absolute",
                "edge_feather": 0,
                "temporal_stability": 1.0,
                "color_tolerance": "strict"
            },
            "deployment_info": {
                "generated_on": "mac",
                "config_hash": hashlib.md5(str(metadata).encode()).hexdigest()[:12]
            }
        }
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"   ✅ Config saved to: {output_path}")
        
        return config


def main():
    """Test product preservation logic"""
    print("\n🛡️ Testing Product Preservation Logic (Mac)")
    print("="*50)
    
    # Initialize preservation logic
    logic = ProductPreservationLogic()
    
    # Test with existing product image
    test_image_path = "test_product_watch.png"
    
    if Path(test_image_path).exists():
        print(f"\n1. Testing with real product: {test_image_path}")
        config = logic.generate_preservation_config(
            product_image_path=test_image_path,
            output_path="product_preservation_config.json"
        )
        
        if config:
            print(f"   ✅ Generated config with {len(config['pixel_locks'])} lock rules")
            print(f"   ✅ Preservation coverage: {config['metadata']['coverage_percent']:.1f}%")
            print(f"   ✅ Confidence: {config['metadata']['confidence']:.3f}")
    else:
        # Create test image
        print("\n1. Creating test product image...")
        test_img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
        
        # Add red circle (product)
        center = (100, 100)
        radius = 50
        y, x = np.ogrid[:200, :200]
        mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        test_img[mask_circle] = [255, 0, 0]  # Red product
        
        # Save test image
        test_pil = Image.fromarray(test_img)
        test_pil.save("test_simple_product.png")
        
        # Test with created image
        config = logic.generate_preservation_config(
            product_image_path="test_simple_product.png",
            output_path="test_preservation_config.json"
        )
        
        if config:
            print(f"   ✅ Test config generated")
            print(f"   ✅ Lock rules: {len(config['pixel_locks'])}")
    
    print("\n2. Testing preservation strength calculation...")
    for frame in [0, 15, 30]:
        strength = logic.calculate_preservation_strength(
            frame_index=frame,
            total_frames=30,
            movement_speed=0.2
        )
        print(f"   Frame {frame}: Strength {strength:.3f}")
    
    print("\n✅ Product Preservation Logic Test Complete!")
    print("🛡️ Ready for zero-hallucination deployment")


if __name__ == "__main__":
    main()