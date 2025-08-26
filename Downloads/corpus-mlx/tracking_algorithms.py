#!/usr/bin/env python3
"""
Product Tracking Algorithms
Frame-to-frame tracking mathematics and logic
NO actual video processing - just the algorithms (Mac development)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum
import cv2
from PIL import Image
import json
import hashlib
from pathlib import Path
import math

class TrackingMethod(Enum):
    """Different tracking approaches"""
    OPTICAL_FLOW = "optical_flow"
    CENTROID_TRACKING = "centroid_tracking"
    KALMAN_FILTER = "kalman_filter" 
    CORRELATION_TRACKING = "correlation_tracking"
    FEATURE_MATCHING = "feature_matching"
    COMBINED = "combined"


class TrackingState(Enum):
    """Tracking state for each product"""
    DETECTED = "detected"        # Product found and tracked
    PREDICTED = "predicted"      # Position predicted (temporary occlusion)
    LOST = "lost"               # Tracking lost
    RECOVERED = "recovered"      # Tracking recovered after loss


@dataclass
class ProductTrack:
    """Individual product tracking state"""
    track_id: int
    current_bbox: Tuple[int, int, int, int]
    current_centroid: Tuple[float, float]
    velocity: Tuple[float, float]  # dx/dt, dy/dt
    confidence: float
    state: TrackingState
    age: int  # Number of frames tracked
    missed_frames: int  # Consecutive frames where detection failed
    history: List[Dict]  # Historical positions and metadata
    
    def update_position(
        self,
        new_bbox: Tuple[int, int, int, int],
        confidence: float,
        frame_index: int
    ):
        """Update track with new detection"""
        # Calculate new centroid
        new_centroid = (
            (new_bbox[0] + new_bbox[2]) / 2,
            (new_bbox[1] + new_bbox[3]) / 2
        )
        
        # Update velocity
        if self.current_centroid:
            self.velocity = (
                new_centroid[0] - self.current_centroid[0],
                new_centroid[1] - self.current_centroid[1]
            )
        
        # Store history
        self.history.append({
            "frame": frame_index,
            "bbox": self.current_bbox,
            "centroid": self.current_centroid,
            "velocity": self.velocity,
            "confidence": self.confidence,
            "state": self.state.value
        })
        
        # Update current state
        self.current_bbox = new_bbox
        self.current_centroid = new_centroid
        self.confidence = confidence
        self.state = TrackingState.DETECTED
        self.age += 1
        self.missed_frames = 0
    
    def predict_next_position(self) -> Tuple[int, int, int, int]:
        """Predict next position based on velocity"""
        if not self.current_centroid or not self.velocity:
            return self.current_bbox
        
        # Predict centroid
        pred_x = self.current_centroid[0] + self.velocity[0]
        pred_y = self.current_centroid[1] + self.velocity[1]
        
        # Calculate predicted bbox (maintain size)
        w = self.current_bbox[2] - self.current_bbox[0]
        h = self.current_bbox[3] - self.current_bbox[1]
        
        pred_bbox = (
            int(pred_x - w/2),
            int(pred_y - h/2),
            int(pred_x + w/2),
            int(pred_y + h/2)
        )
        
        return pred_bbox


class TrackingAlgorithms:
    """
    Advanced tracking algorithms for video sequences
    Mathematics and logic only - NO actual video processing
    """
    
    def __init__(self, max_disappeared: int = 10, max_distance: float = 100):
        self.tracks = {}  # track_id -> ProductTrack
        self.next_track_id = 0
        self.max_disappeared = max_disappeared  # Max frames before declaring lost
        self.max_distance = max_distance       # Max distance for association
        
        # Tracking parameters
        self.params = {
            "association_threshold": 50.0,    # Distance threshold for association
            "velocity_smoothing": 0.3,        # Velocity smoothing factor
            "confidence_threshold": 0.3,      # Minimum confidence for detection
            "prediction_weight": 0.7,         # Weight for predicted vs detected position
            "size_change_threshold": 0.5,     # Maximum size change between frames
            "direction_change_threshold": 45   # Maximum direction change (degrees)
        }
    
    def track_products_in_sequence(
        self,
        detections_sequence: List[List[Dict]],
        frame_rate: float = 30.0
    ) -> Dict[int, ProductTrack]:
        """
        Track products across a sequence of detection frames
        
        Args:
            detections_sequence: List of detections per frame
                Each detection: {"bbox": (x1,y1,x2,y2), "confidence": float, "properties": dict}
            frame_rate: Video frame rate for velocity calculations
        
        Returns:
            Dictionary of track_id -> ProductTrack with complete histories
        """
        print(f"🎬 Tracking products across {len(detections_sequence)} frames")
        print(f"   Frame rate: {frame_rate} fps")
        print(f"   Max tracking distance: {self.max_distance}")
        
        self.tracks = {}
        self.next_track_id = 0
        
        for frame_idx, frame_detections in enumerate(detections_sequence):
            self._process_frame(frame_detections, frame_idx, frame_rate)
            
            if frame_idx % 10 == 0:  # Progress update every 10 frames
                active_tracks = sum(1 for t in self.tracks.values() if t.state != TrackingState.LOST)
                print(f"   Frame {frame_idx}: {active_tracks} active tracks")
        
        # Final statistics
        final_tracks = len(self.tracks)
        active_tracks = sum(1 for t in self.tracks.values() if t.state != TrackingState.LOST)
        avg_age = np.mean([t.age for t in self.tracks.values()]) if self.tracks else 0
        
        print(f"   ✅ Tracking complete:")
        print(f"      Total tracks created: {final_tracks}")
        print(f"      Active at end: {active_tracks}")
        print(f"      Average track age: {avg_age:.1f} frames")
        
        return self.tracks
    
    def _process_frame(
        self,
        detections: List[Dict],
        frame_index: int,
        frame_rate: float
    ):
        """Process single frame of detections"""
        
        # Filter detections by confidence
        valid_detections = [
            d for d in detections 
            if d.get("confidence", 0) >= self.params["confidence_threshold"]
        ]
        
        if not valid_detections:
            # No detections - update all tracks as missed
            for track in self.tracks.values():
                track.missed_frames += 1
                if track.missed_frames > self.max_disappeared:
                    track.state = TrackingState.LOST
                else:
                    track.state = TrackingState.PREDICTED
            return
        
        # Get current track predictions
        track_predictions = {}
        for track_id, track in self.tracks.items():
            if track.state != TrackingState.LOST:
                track_predictions[track_id] = track.predict_next_position()
        
        # Associate detections with existing tracks
        associations = self._associate_detections_to_tracks(
            valid_detections, track_predictions
        )
        
        # Update existing tracks
        updated_track_ids = set()
        for detection_idx, track_id in associations.items():
            detection = valid_detections[detection_idx]
            self.tracks[track_id].update_position(
                new_bbox=detection["bbox"],
                confidence=detection["confidence"],
                frame_index=frame_index
            )
            updated_track_ids.add(track_id)
        
        # Handle unassociated tracks (missed detections)
        for track_id, track in self.tracks.items():
            if track_id not in updated_track_ids and track.state != TrackingState.LOST:
                track.missed_frames += 1
                if track.missed_frames > self.max_disappeared:
                    track.state = TrackingState.LOST
                else:
                    track.state = TrackingState.PREDICTED
        
        # Create new tracks for unassociated detections
        unassociated_detections = [
            (i, d) for i, d in enumerate(valid_detections)
            if i not in associations
        ]
        
        for det_idx, detection in unassociated_detections:
            self._create_new_track(detection, frame_index)
    
    def _associate_detections_to_tracks(
        self,
        detections: List[Dict],
        track_predictions: Dict[int, Tuple[int, int, int, int]]
    ) -> Dict[int, int]:
        """
        Associate detections to existing tracks using Hungarian algorithm approach
        
        Returns:
            Dictionary mapping detection_index -> track_id
        """
        if not detections or not track_predictions:
            return {}
        
        # Calculate cost matrix (distances)
        cost_matrix = []
        track_ids = list(track_predictions.keys())
        
        for detection in detections:
            detection_centroid = self._bbox_to_centroid(detection["bbox"])
            row_costs = []
            
            for track_id in track_ids:
                predicted_bbox = track_predictions[track_id]
                predicted_centroid = self._bbox_to_centroid(predicted_bbox)
                
                # Calculate association cost
                cost = self._calculate_association_cost(
                    detection, detection_centroid,
                    self.tracks[track_id], predicted_centroid
                )
                row_costs.append(cost)
            
            cost_matrix.append(row_costs)
        
        if not cost_matrix:
            return {}
        
        # Simple greedy assignment (could use Hungarian for optimal)
        associations = self._greedy_assignment(
            cost_matrix, track_ids, self.params["association_threshold"]
        )
        
        return associations
    
    def _calculate_association_cost(
        self,
        detection: Dict,
        detection_centroid: Tuple[float, float],
        track: ProductTrack,
        predicted_centroid: Tuple[float, float]
    ) -> float:
        """
        Calculate cost of associating detection with track
        Lower cost = better match
        """
        # Distance cost
        distance = math.sqrt(
            (detection_centroid[0] - predicted_centroid[0]) ** 2 +
            (detection_centroid[1] - predicted_centroid[1]) ** 2
        )
        distance_cost = distance / self.max_distance
        
        # Size consistency cost
        det_size = self._bbox_size(detection["bbox"])
        track_size = self._bbox_size(track.current_bbox)
        size_ratio = abs(det_size - track_size) / max(det_size, track_size)
        size_cost = min(1.0, size_ratio / self.params["size_change_threshold"])
        
        # Confidence cost (prefer high confidence detections)
        confidence_cost = 1.0 - detection.get("confidence", 0.5)
        
        # Velocity consistency cost
        velocity_cost = 0.0
        if track.velocity and len(track.history) > 1:
            # Check if movement direction is consistent
            expected_next = (
                track.current_centroid[0] + track.velocity[0],
                track.current_centroid[1] + track.velocity[1]
            )
            velocity_distance = math.sqrt(
                (detection_centroid[0] - expected_next[0]) ** 2 +
                (detection_centroid[1] - expected_next[1]) ** 2
            )
            velocity_cost = min(1.0, velocity_distance / self.max_distance)
        
        # Weighted combination
        total_cost = (
            distance_cost * 0.4 +      # 40% distance
            size_cost * 0.2 +          # 20% size consistency
            confidence_cost * 0.2 +    # 20% confidence
            velocity_cost * 0.2        # 20% velocity consistency
        )
        
        return total_cost
    
    def _greedy_assignment(
        self,
        cost_matrix: List[List[float]],
        track_ids: List[int],
        threshold: float
    ) -> Dict[int, int]:
        """
        Greedy assignment of detections to tracks
        """
        associations = {}
        used_tracks = set()
        
        # Convert to numpy for easier manipulation
        costs = np.array(cost_matrix)
        
        # Find assignments greedily
        for _ in range(min(len(costs), len(track_ids))):
            if costs.size == 0:
                break
            
            # Find minimum cost
            min_idx = np.unravel_index(np.argmin(costs), costs.shape)
            min_cost = costs[min_idx]
            
            if min_cost > threshold:
                break  # No more good associations
            
            detection_idx, track_idx = min_idx
            track_id = track_ids[track_idx]
            
            # Make association
            associations[detection_idx] = track_id
            used_tracks.add(track_id)
            
            # Remove this detection and track from consideration
            costs = np.delete(costs, detection_idx, axis=0)
            costs = np.delete(costs, track_idx, axis=1)
            
            # Update track_ids list
            track_ids = [tid for i, tid in enumerate(track_ids) if i != track_idx]
        
        return associations
    
    def _create_new_track(self, detection: Dict, frame_index: int):
        """Create new track for unassociated detection"""
        track = ProductTrack(
            track_id=self.next_track_id,
            current_bbox=detection["bbox"],
            current_centroid=self._bbox_to_centroid(detection["bbox"]),
            velocity=(0.0, 0.0),
            confidence=detection["confidence"],
            state=TrackingState.DETECTED,
            age=1,
            missed_frames=0,
            history=[]
        )
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
    
    def _bbox_to_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Convert bounding box to centroid"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _bbox_size(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate bounding box size (area)"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def calculate_tracking_quality_metrics(
        self,
        tracks: Dict[int, ProductTrack]
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for tracking performance
        """
        if not tracks:
            return {"error": "no_tracks"}
        
        # Basic statistics
        total_tracks = len(tracks)
        active_tracks = sum(1 for t in tracks.values() if t.state != TrackingState.LOST)
        
        # Track lengths
        track_lengths = [t.age for t in tracks.values()]
        avg_track_length = np.mean(track_lengths)
        max_track_length = np.max(track_lengths)
        
        # Confidence statistics
        confidences = []
        for track in tracks.values():
            track_confidences = [h.get("confidence", 0) for h in track.history]
            confidences.extend(track_confidences)
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Velocity statistics (measure of smoothness)
        velocities = []
        for track in tracks.values():
            track_velocities = [
                math.sqrt(v[0]**2 + v[1]**2) 
                for h in track.history 
                if h.get("velocity") 
                for v in [h["velocity"]]
            ]
            velocities.extend(track_velocities)
        
        avg_velocity = np.mean(velocities) if velocities else 0.0
        velocity_std = np.std(velocities) if velocities else 0.0
        
        # Tracking stability (lower is better)
        stability_scores = []
        for track in tracks.values():
            if len(track.history) > 1:
                # Calculate position variance
                positions = [(h["centroid"][0], h["centroid"][1]) for h in track.history if h.get("centroid")]
                if len(positions) > 1:
                    x_positions = [p[0] for p in positions]
                    y_positions = [p[1] for p in positions]
                    position_variance = np.var(x_positions) + np.var(y_positions)
                    stability_scores.append(position_variance)
        
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        return {
            "total_tracks": total_tracks,
            "active_tracks": active_tracks,
            "track_retention_rate": active_tracks / max(1, total_tracks),
            "avg_track_length": avg_track_length,
            "max_track_length": max_track_length,
            "avg_confidence": avg_confidence,
            "avg_velocity": avg_velocity,
            "velocity_stability": 1.0 / (1.0 + velocity_std),  # Higher is more stable
            "position_stability": 1.0 / (1.0 + avg_stability / 1000),  # Normalized stability
            "overall_quality": self._calculate_overall_quality(
                active_tracks / max(1, total_tracks),
                avg_confidence,
                1.0 / (1.0 + velocity_std),
                1.0 / (1.0 + avg_stability / 1000)
            )
        }
    
    def _calculate_overall_quality(
        self,
        retention: float,
        confidence: float,
        velocity_stability: float,
        position_stability: float
    ) -> float:
        """Calculate overall tracking quality score (0-1)"""
        return (
            retention * 0.3 +           # 30% retention
            confidence * 0.3 +          # 30% detection confidence
            velocity_stability * 0.2 +   # 20% velocity consistency
            position_stability * 0.2     # 20% position stability
        )
    
    def generate_tracking_schedule(
        self,
        video_metadata: Dict,
        expected_products: int = 1
    ) -> Dict:
        """
        Generate tracking schedule for deployment to RunPod
        """
        print(f"📋 Generating tracking schedule...")
        print(f"   Expected products: {expected_products}")
        print(f"   Video frames: {video_metadata.get('frame_count', 'unknown')}")
        
        schedule = {
            "version": "1.0",
            "tracking_config": {
                "max_products": expected_products * 2,  # Allow some flexibility
                "tracking_method": "combined",
                "parameters": self.params.copy(),
                "quality_thresholds": {
                    "min_track_length": 5,        # Minimum frames to consider valid
                    "min_confidence": 0.3,        # Minimum detection confidence
                    "max_missed_frames": self.max_disappeared,
                    "association_threshold": self.params["association_threshold"]
                }
            },
            "video_metadata": video_metadata,
            "processing_instructions": {
                "frame_by_frame": True,
                "save_intermediate": False,  # Save space on RunPod
                "track_histories": True,     # Keep full tracking history
                "quality_metrics": True      # Calculate quality metrics
            },
            "output_format": {
                "tracks": "json",           # Track data format
                "visualizations": "png",    # Debug visualizations
                "metrics": "json"           # Quality metrics
            },
            "deployment_info": {
                "generated_on": "mac",
                "algorithm_version": "1.0",
                "config_hash": hashlib.md5(
                    json.dumps(self.params, sort_keys=True).encode()
                ).hexdigest()[:12]
            }
        }
        
        print(f"   ✅ Schedule generated with hash: {schedule['deployment_info']['config_hash']}")
        return schedule


def main():
    """Test tracking algorithms"""
    print("\n🎬 Testing Product Tracking Algorithms (Mac)")
    print("="*55)
    
    # Initialize tracker
    tracker = TrackingAlgorithms(max_disappeared=5, max_distance=80)
    
    # Generate synthetic detection sequence for testing
    print("\n1. Creating synthetic detection sequence...")
    
    # Simulate 3 products moving across frames
    frame_count = 30
    detections_sequence = []
    
    for frame_idx in range(frame_count):
        frame_detections = []
        
        # Product 1: Moving right
        if frame_idx < 25:  # Disappears near end
            x1 = 50 + frame_idx * 8
            y1 = 100 + int(5 * math.sin(frame_idx * 0.2))  # Slight vertical oscillation
            bbox1 = (x1, y1, x1 + 60, y1 + 80)
            frame_detections.append({
                "bbox": bbox1,
                "confidence": 0.8 + 0.1 * math.sin(frame_idx * 0.1),
                "properties": {"color": "red", "size": "medium"}
            })
        
        # Product 2: Moving diagonally
        if frame_idx > 5:  # Starts after frame 5
            x2 = 200 + frame_idx * 3
            y2 = 50 + frame_idx * 4
            bbox2 = (x2, y2, x2 + 40, y2 + 60)
            frame_detections.append({
                "bbox": bbox2,
                "confidence": 0.7 + 0.05 * frame_idx / 30,
                "properties": {"color": "blue", "size": "small"}
            })
        
        # Product 3: Stationary with occasional occlusion
        if frame_idx % 4 != 0:  # Missing every 4th frame (occlusion)
            bbox3 = (150, 200, 190, 250)
            frame_detections.append({
                "bbox": bbox3,
                "confidence": 0.6 + 0.2 * math.cos(frame_idx * 0.3),
                "properties": {"color": "green", "size": "large"}
            })
        
        detections_sequence.append(frame_detections)
    
    print(f"   ✅ Created {frame_count} frames of synthetic data")
    print(f"   ✅ Average detections per frame: {np.mean([len(f) for f in detections_sequence]):.1f}")
    
    # Run tracking
    print("\n2. Running tracking algorithms...")
    tracks = tracker.track_products_in_sequence(detections_sequence, frame_rate=30.0)
    
    # Analyze results
    print(f"\n3. Analyzing tracking results...")
    for track_id, track in tracks.items():
        print(f"   Track {track_id}:")
        print(f"     State: {track.state.value}")
        print(f"     Age: {track.age} frames")
        print(f"     Missed: {track.missed_frames} frames")
        print(f"     Final confidence: {track.confidence:.3f}")
        if track.velocity:
            speed = math.sqrt(track.velocity[0]**2 + track.velocity[1]**2)
            print(f"     Speed: {speed:.1f} pixels/frame")
    
    # Calculate quality metrics
    print("\n4. Calculating quality metrics...")
    metrics = tracker.calculate_tracking_quality_metrics(tracks)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.3f}")
        else:
            print(f"   {metric}: {value}")
    
    # Generate tracking schedule
    print("\n5. Generating deployment schedule...")
    video_metadata = {
        "frame_count": frame_count,
        "fps": 30.0,
        "resolution": [640, 480],
        "expected_products": 3
    }
    
    schedule = tracker.generate_tracking_schedule(video_metadata, expected_products=3)
    
    # Save schedule
    with open("tracking_schedule.json", 'w') as f:
        json.dump(schedule, f, indent=2)
    print("   ✅ Schedule saved to: tracking_schedule.json")
    
    print("\n✅ Product Tracking Algorithms Test Complete!")
    print("🎬 Ready for video sequence processing on RunPod")


if __name__ == "__main__":
    main()