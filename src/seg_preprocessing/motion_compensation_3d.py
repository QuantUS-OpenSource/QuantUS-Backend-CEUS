"""
3D Motion Compensation for CEUS Analysis
Based on ILSA tracking
Axis convention: (X, Y, Z, T) = (lateral, depth, elevational, time)
"""

import numpy as np
from scipy.ndimage import shift
from scipy.signal import correlate
from typing import Tuple, List, Optional
from dataclasses import dataclass
from collections import Counter
import cv2
from tqdm import tqdm

@dataclass
class MotionCompensationResult:
    """Store motion compensation results efficiently"""
    translation_vectors: np.ndarray
    reference_frame: int
    correlations: np.ndarray
    reference_bbox: 'BoundingBox3D'
    tracked_bboxes: List['BoundingBox3D']
    
    def get_translation(self, frame_idx: int) -> Tuple[float, float, float]:
        return tuple(self.translation_vectors[frame_idx])
    
    def apply_to_mask(self, mask_3d: np.ndarray, frame_idx: int, order: int = 0) -> np.ndarray:
        dx, dy, dz = self.get_translation(frame_idx)
        return shift(mask_3d, shift=[dx, dy, dz], order=order, cval=0,
                    prefilter=True if order > 0 else False)
    
    def apply_to_all_frames(self, mask_3d: np.ndarray, order: int = 0) -> np.ndarray:
        n_frames = len(self.translation_vectors)
        mc_mask_4d = np.zeros((*mask_3d.shape, n_frames), dtype=mask_3d.dtype)
        for frame_idx in range(n_frames):
            mc_mask_4d[..., frame_idx] = self.apply_to_mask(mask_3d, frame_idx, order)
        return mc_mask_4d

@dataclass
class BoundingBox3D:
    """3D Bounding box definition"""
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min)
    
    @property
    def center(self) -> Tuple[float, float, float]:
        return ((self.x_min + self.x_max) / 2,
                (self.y_min + self.y_max) / 2,
                (self.z_min + self.z_max) / 2)
    
    def expand(self, margin: Tuple[int, int, int]) -> 'BoundingBox3D':
        return BoundingBox3D(
            max(0, self.x_min - margin[0]), self.x_max + margin[0],
            max(0, self.y_min - margin[1]), self.y_max + margin[1],
            max(0, self.z_min - margin[2]), self.z_max + margin[2])
    
    def extract_from_volume(self, volume: np.ndarray) -> np.ndarray:
        return volume[self.x_min:self.x_max, self.y_min:self.y_max, self.z_min:self.z_max]
    
    def translate(self, dx: int, dy: int, dz: int) -> 'BoundingBox3D':
        return BoundingBox3D(
            self.x_min + dx, self.x_max + dx,
            self.y_min + dy, self.y_max + dy,
            self.z_min + dz, self.z_max + dz)
    
    @classmethod
    def from_mask(cls, mask: np.ndarray, padding: int = 0) -> 'BoundingBox3D':
        nonzero = np.argwhere(mask > 0)
        if len(nonzero) == 0:
            raise ValueError("Mask contains no non-zero voxels")
        
        x_min = max(0, nonzero[:, 0].min() - padding)
        x_max = min(mask.shape[0], nonzero[:, 0].max() + 1 + padding)
        y_min = max(0, nonzero[:, 1].min() - padding)
        y_max = min(mask.shape[1], nonzero[:, 1].max() + 1 + padding)
        z_min = max(0, nonzero[:, 2].min() - padding)
        z_max = min(mask.shape[2], nonzero[:, 2].max() + 1 + padding)
        
        return cls(x_min, x_max, y_min, y_max, z_min, z_max)


class MotionCompensation3D:
    """
    3D Motion Compensation using ILSA tracking or Reference-only tracking
    """
    
    def __init__(
        self, 
        search_margin_ratio: float = 0.5 / 30,
        use_reference_only: bool = False  # ← NEW PARAMETER
    ):
        """
        Args:
            search_margin_ratio: Search margin as ratio of volume dimensions
            use_reference_only: If True, always track from reference frame only
                               If False, use ILSA (compare reference vs previous)
        """
        self.search_margin_ratio = search_margin_ratio
        self.use_reference_only = use_reference_only
    
    def compute_search_margin(self, volume_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(int(self.search_margin_ratio * x) for x in volume_shape)
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        mean = np.mean(volume)
        std = np.std(volume)
        if std == 0:
            return volume - mean
        return (volume - mean) / std
       
    def compute_3d_correlation_vectorized(
        self,
        volumes: np.ndarray,
        reference_voi: np.ndarray,
        search_bbox: BoundingBox3D
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized 3D correlation computation"""
        n_frames = volumes.shape[-1]
        ref_shape = reference_voi.shape
        ref_normalized = self.normalize_volume(reference_voi)
        
        search_shape = search_bbox.shape
        corr_shape = tuple(s - r + 1 for s, r in zip(search_shape, ref_shape))
        correlation_map = np.zeros((n_frames, *corr_shape))
        
        search_region = search_bbox.extract_from_volume(volumes[..., 0])
        search_normalized = self.normalize_volume(search_region)
        
        correlation = correlate(search_normalized, ref_normalized, mode='valid', method='fft')
        
        ref_sum_sq = np.sum(ref_normalized ** 2)
        search_sq = search_normalized ** 2
        kernel = np.ones(ref_shape)
        local_sum_sq = correlate(search_sq, kernel, mode='valid', method='fft')
        
        denominator = np.sqrt(ref_sum_sq * local_sum_sq)
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        correlation_map = correlation / denominator
        max_correlations = np.max(correlation_map.reshape(n_frames, -1), axis=1)
        
        return correlation_map, max_correlations
    
    def find_optimal_translation(
        self,
        correlation_map: np.ndarray,
        search_bbox: BoundingBox3D,
        reference_bbox: BoundingBox3D
    ) -> Tuple[int, int, int]:
        """Find optimal translation from correlation map"""
        max_idx = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
        
        dx = search_bbox.x_min + max_idx[0] - reference_bbox.x_min
        dy = search_bbox.y_min + max_idx[1] - reference_bbox.y_min
        dz = search_bbox.z_min + max_idx[2] - reference_bbox.z_min
        
        return dx, dy, dz
    
    def track_motion_ilsa_3d(
        self,
        volumes: np.ndarray,
        reference_frame_idx: int,
        reference_bbox: BoundingBox3D
    ) -> Tuple[List[BoundingBox3D], List[float]]:
        """
        ILSA tracking with optional reference-only mode
        
        Args:
            volumes: All volume frames (X, Y, Z, T), lateral, depth, elevational, time
            reference_frame_idx: Index of reference frame
            reference_bbox: Bounding box around lesion in reference frame
            
        Returns:
            tracked_bboxes: List of bounding boxes for each frame
            correlations: List of correlation values
        """
        n_frames = volumes.shape[-1]
        ref_img = volumes[..., reference_frame_idx]
        
        # Apply image enhancement
        ref_voi = reference_bbox.extract_from_volume(ref_img)
        
        search_margin = self.compute_search_margin(volumes.shape[:-1])
        
        tracked_bboxes = [None] * n_frames
        correlations = [0.0] * n_frames
        tracking_sources = [''] * n_frames
        
        # Set reference frame
        tracked_bboxes[reference_frame_idx] = reference_bbox
        correlations[reference_frame_idx] = 1.0
        tracking_sources[reference_frame_idx] = 'reference'
        
        # Print tracking mode
        if self.use_reference_only:
            print("\n=== Reference-Only Tracking Mode ===")
            print("All frames will be tracked from reference frame only")
        else:
            print("\n=== ILSA Tracking Mode ===")
            print("Comparing reference frame vs previous frame")
        
        # === FORWARD TRACKING ===
        forward_frames = range(reference_frame_idx + 1, n_frames)
        forward_frames = tqdm(forward_frames, desc="Tracking frames", unit="frame")
        
        for frame_idx in forward_frames:
            prev_bbox = tracked_bboxes[frame_idx - 1]
            search_bbox = prev_bbox.expand(search_margin)
            
            # Extract and enhance current frame
            image_analysis = volumes[..., frame_idx]
            final_image = image_analysis[..., np.newaxis]
            
            # Always compute reference correlation
            corr_map_ref, max_corr_ref = self.compute_3d_correlation_vectorized(
                final_image, ref_voi, search_bbox
            )
            
            if self.use_reference_only:
                # ★ REFERENCE-ONLY MODE: Skip previous frame comparison
                dx, dy, dz = self.find_optimal_translation(
                    corr_map_ref, search_bbox, reference_bbox
                )
                tracked_bboxes[frame_idx] = reference_bbox.translate(dx, dy, dz)
                correlations[frame_idx] = max_corr_ref[0]
                tracking_sources[frame_idx] = 'reference_only'
                
            else:
                # ★ ILSA MODE: Compare reference vs previous
                prev_voi = prev_bbox.extract_from_volume(volumes[..., frame_idx - 1])
                corr_map_prev, max_corr_prev = self.compute_3d_correlation_vectorized(
                    final_image, prev_voi, search_bbox
                )
                
                # Pick whichever has better correlation
                if max_corr_ref[0] >= max_corr_prev[0]:
                    dx, dy, dz = self.find_optimal_translation(
                        corr_map_ref, search_bbox, reference_bbox
                    )
                    tracked_bboxes[frame_idx] = reference_bbox.translate(dx, dy, dz)
                    correlations[frame_idx] = max_corr_ref[0]
                    tracking_sources[frame_idx] = 'reference'
                else:
                    dx, dy, dz = self.find_optimal_translation(
                        corr_map_prev, search_bbox, prev_bbox
                    )
                    tracked_bboxes[frame_idx] = prev_bbox.translate(dx, dy, dz)
                    correlations[frame_idx] = max_corr_prev[0]
                    tracking_sources[frame_idx] = 'previous'
        
        # === PRINT STATS ===
        source_counts = Counter(tracking_sources)
        print(f"\n=== Tracking Complete ===")
        print(f"Sources: {dict(source_counts)}")
        print(f"Mean correlation: {np.mean(correlations):.3f}")
        print(f"Min correlation: {np.min(correlations):.3f}")
        
        return tracked_bboxes, correlations
    
class OpticalFlowMotionCompensation3D:
    """
    3D Motion Compensation using TRUE 3D Feature Tracking
    Axis convention: (X, Y, Z, T) = (lateral, depth, elevational, time)
    
    This implementation tracks features in full 3D using volumetric correlation:
    1. Detect good features to track in the reference frame (across multiple slices)
    2. For each feature, extract a 3D patch around it
    3. Search in 3D neighborhood in next frame using normalized cross-correlation
    4. Track motion in all three dimensions: (dx, dy, dz)
    5. Estimate global 3D translation from all tracked features
    
    Key advantage: Captures true 3D motion including Z-axis (elevational) displacement
    
    Note: Unlike 2D ultrasound, we do NOT reject frames for out-of-plane motion
    since we have full 3D volumes. All frames can be tracked and compensated.
    """
    
    def __init__(
        self,
        feature_params: Optional[dict] = None,
        min_features_ratio: float = 0.2,
        patch_size_x: int = 7,
        patch_size_y: int = 7,
        patch_size_z: int = 5,
        search_range_x: int = 7,
        search_range_y: int = 7,
        search_range_z: int = 3,
        correlation_threshold: float = 0.5
    ):
        """
        Initialize optical flow motion compensation
        
        Args:
            feature_params: Parameters for good features to track
            min_features_ratio: Minimum ratio of tracked features (for confidence)
            patch_size_x: Half-size of patch in X direction (lateral)
            patch_size_y: Half-size of patch in Y direction (depth)
            patch_size_z: Half-size of patch in Z direction (elevational)
            search_range_x: Search range in X direction
            search_range_y: Search range in Y direction
            search_range_z: Search range in Z direction
            correlation_threshold: Minimum correlation to accept match
        """
        # Shi-Tomasi corner detection parameters
        self.feature_params = feature_params or {
            'maxCorners': 100,
            'qualityLevel': 0.3,
            'minDistance': 7,
            'blockSize': 7
        }
        
        self.min_features_ratio = min_features_ratio
        
        # 3D tracking parameters (x, y, z)
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.patch_size_z = patch_size_z
        self.search_range_x = search_range_x
        self.search_range_y = search_range_y
        self.search_range_z = search_range_z
        self.correlation_threshold = correlation_threshold
    
    def detect_features_3d(
        self,
        volume: np.ndarray,  # (X, Y, Z)
        bbox: BoundingBox3D
    ) -> List[np.ndarray]:
        """
        Detect good features to track in 3D volume within bbox
        
        Strategy: Detect features on multiple elevational (Z) slices
        
        Args:
            volume: 3D volume (X, Y, Z) = (lateral, depth, elevational)
            bbox: Bounding box defining ROI
            
        Returns:
            List of feature points [(N, 3)] where each point is (x, y, z)
        """
        roi = bbox.extract_from_volume(volume)
        width, height, depth = roi.shape  # (X, Y, Z)
        
        # Select slices to detect features (every few slices to avoid redundancy)
        slice_indices = range(0, depth, max(1, depth // 10))
        
        all_features = []
        
        for z_idx in slice_indices:
            # Get slice and normalize - need to transpose for cv2
            # cv2 expects (height, width) which is (Y, X)
            slice_img = roi[:, :, z_idx].T  # Transpose to (Y, X)
            slice_img = cv2.normalize(
                slice_img, None, 0, 255, 
                cv2.NORM_MINMAX
            ).astype(np.uint8)
            
            # Detect corners
            corners = cv2.goodFeaturesToTrack(
                slice_img,
                mask=None,
                **self.feature_params
            )
            
            if corners is not None:
                # Convert to 3D coordinates (absolute coordinates in volume)
                # cv2.goodFeaturesToTrack returns corners as (x, y) in image coordinates
                # but since we transposed, it's returning positions in (Y, X) space
                corners_3d = np.zeros((len(corners), 3))
                corners_3d[:, 0] = corners[:, 0, 0] + bbox.x_min  # Absolute X (was col in transposed)
                corners_3d[:, 1] = corners[:, 0, 1] + bbox.y_min  # Absolute Y (was row in transposed)
                corners_3d[:, 2] = z_idx + bbox.z_min  # Absolute Z
                
                all_features.append(corners_3d)
        
        return all_features
    
    def track_features_3d(
        self,
        prev_volume: np.ndarray,  # (X, Y, Z)
        curr_volume: np.ndarray,  # (X, Y, Z)
        prev_features: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Track features in TRUE 3D using volumetric correlation
        
        Strategy:
        1. For each feature at (x, y, z) in prev_volume
        2. Extract 3D patch around it
        3. Search in 3D neighborhood in curr_volume
        4. Find best match using 3D normalized cross-correlation
        
        Args:
            prev_volume: Previous frame volume (X, Y, Z)
            curr_volume: Current frame volume (X, Y, Z)
            prev_features: List of feature points from previous frame
            
        Returns:
            curr_features: Tracked features in current frame
            status_list: Status of each feature (1=good, 0=lost)
            motion_vectors: Motion vectors for each feature (dx, dy, dz)
        """
        # Flatten all features
        all_prev_features = []
        for feat_array in prev_features:
            if len(feat_array) > 0:
                all_prev_features.append(feat_array)
        
        if len(all_prev_features) == 0:
            return [], [], []
        
        all_prev_features = np.vstack(all_prev_features)
        
        tracked_features = []
        tracked_status = []
        tracked_motion = []
        
        width, height, depth = prev_volume.shape  # (X, Y, Z)
        
        for feat in all_prev_features:
            x, y, z = feat.astype(int)
            
            # Check bounds for patch extraction
            if (x - self.patch_size_x < 0 or x + self.patch_size_x >= width or
                y - self.patch_size_y < 0 or y + self.patch_size_y >= height or
                z - self.patch_size_z < 0 or z + self.patch_size_z >= depth):
                # Feature too close to boundary
                tracked_features.append(feat)
                tracked_status.append(0)  # Lost
                tracked_motion.append(np.array([0.0, 0.0, 0.0]))
                continue
            
            # Extract 3D patch from previous volume
            prev_patch = prev_volume[
                x - self.patch_size_x : x + self.patch_size_x + 1,
                y - self.patch_size_y : y + self.patch_size_y + 1,
                z - self.patch_size_z : z + self.patch_size_z + 1
            ]
            
            # Normalize patch
            prev_mean = np.mean(prev_patch)
            prev_std = np.std(prev_patch)
            if prev_std < 1e-6:
                # Homogeneous patch, skip
                tracked_features.append(feat)
                tracked_status.append(0)
                tracked_motion.append(np.array([0.0, 0.0, 0.0]))
                continue
                
            prev_patch_norm = (prev_patch - prev_mean) / prev_std
            
            # Define search region in current volume
            x_min = max(self.patch_size_x, x - self.search_range_x)
            x_max = min(width - self.patch_size_x, x + self.search_range_x + 1)
            y_min = max(self.patch_size_y, y - self.search_range_y)
            y_max = min(height - self.patch_size_y, y + self.search_range_y + 1)
            z_min = max(self.patch_size_z, z - self.search_range_z)
            z_max = min(depth - self.patch_size_z, z + self.search_range_z + 1)
            
            # Search for best match in 3D
            best_corr = -1
            best_pos = (x, y, z)
            
            for x_search in range(x_min, x_max):
                for y_search in range(y_min, y_max):
                    for z_search in range(z_min, z_max):
                        # Extract candidate patch
                        curr_patch = curr_volume[
                            x_search - self.patch_size_x : x_search + self.patch_size_x + 1,
                            y_search - self.patch_size_y : y_search + self.patch_size_y + 1,
                            z_search - self.patch_size_z : z_search + self.patch_size_z + 1
                        ]
                        
                        # Normalize
                        curr_mean = np.mean(curr_patch)
                        curr_std = np.std(curr_patch)
                        if curr_std < 1e-6:
                            continue
                            
                        curr_patch_norm = (curr_patch - curr_mean) / curr_std
                        
                        # Compute 3D normalized cross-correlation
                        corr = np.sum(prev_patch_norm * curr_patch_norm) / prev_patch_norm.size
                        
                        if corr > best_corr:
                            best_corr = corr
                            best_pos = (x_search, y_search, z_search)
            
            # Check if tracking was successful
            if best_corr > self.correlation_threshold:
                tracked_features.append(np.array(best_pos, dtype=float))
                tracked_status.append(1)  # Good
                motion = np.array(best_pos, dtype=float) - feat
                tracked_motion.append(motion)
            else:
                # Lost tracking
                tracked_features.append(feat)
                tracked_status.append(0)  # Lost
                tracked_motion.append(np.array([0.0, 0.0, 0.0]))
        
        # Return as lists for consistency
        if len(tracked_features) > 0:
            curr_features = [np.array(tracked_features)]
            status_list = [np.array(tracked_status)]
            motion_vectors = [np.array(tracked_motion)]
        else:
            curr_features = []
            status_list = []
            motion_vectors = []
        
        return curr_features, status_list, motion_vectors
    
    def estimate_global_motion(
        self,
        motion_vectors: List[np.ndarray],
        status_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate global 3D translation from tracked features
        
        Uses robust estimation (median + MAD) to find consensus motion
        
        Args:
            motion_vectors: List of motion vectors for each feature (dx, dy, dz)
            status_list: List of status for each feature
            
        Returns:
            global_motion: (dx, dy, dz) translation vector
            confidence: Confidence score (ratio of inliers)
        """
        if not motion_vectors or not status_list:
            return np.zeros(3), 0.0
        
        # Check if lists are empty
        if len(motion_vectors) == 0 or len(status_list) == 0:
            return np.zeros(3), 0.0
        
        # Concatenate all motion vectors
        all_motions = np.vstack(motion_vectors)
        all_status = np.hstack(status_list)
        
        # Filter by status
        valid_motions = all_motions[all_status == 1]
        
        if len(valid_motions) == 0:
            return np.zeros(3), 0.0
        
        # If very few features, just use mean
        if len(valid_motions) < 3:
            return np.mean(valid_motions, axis=0), len(valid_motions) / max(len(all_status), 1)
        
        # Remove outliers using median absolute deviation
        median_motion = np.median(valid_motions, axis=0)
        mad = np.median(np.abs(valid_motions - median_motion), axis=0)
        
        # Avoid division by zero
        mad = np.where(mad == 0, 1e-6, mad)
        
        # Threshold for inliers (3 * MAD)
        threshold = 3 * mad
        inlier_mask = np.all(
            np.abs(valid_motions - median_motion) <= threshold,
            axis=1
        )
        
        inlier_motions = valid_motions[inlier_mask]
        
        if len(inlier_motions) == 0:
            # Fallback to median if no inliers
            return median_motion, len(valid_motions) / max(len(all_status), 1)
        
        # Estimate global motion as mean of inliers
        global_motion = np.mean(inlier_motions, axis=0)
        confidence = len(inlier_motions) / max(len(valid_motions), 1)
        
        return global_motion, confidence
    
    def track_motion(
        self,
        volumes: np.ndarray,  # (X, Y, Z, T)
        reference_frame_idx: int,
        reference_bbox: BoundingBox3D
    ) -> Tuple[List[BoundingBox3D], List[float]]:
        """
        Track motion across frames using 3D optical flow
        
        Unlike 2D ultrasound, we track ALL frames since we have full 3D volumes.
        We don't reject frames - instead we provide confidence scores.
        
        Args:
            volumes: All volume frames (X, Y, Z, T) = (lateral, depth, elevational, time)
            reference_frame_idx: Index of reference frame
            reference_bbox: Bounding box around lesion in reference frame
            
        Returns:
            bboxes: List of bounding boxes for each frame (all valid)
            confidences: List of tracking confidence values (0-1)
        """
        n_frames = volumes.shape[-1]
        
        # Initialize output - ALL frames will have valid bboxes
        tracked_bboxes = [None] * n_frames
        confidences = [0.0] * n_frames
        
        # Reference frame
        tracked_bboxes[reference_frame_idx] = reference_bbox
        confidences[reference_frame_idx] = 1.0
        
        # Detect features in reference frame
        print(f"\nDetecting features in reference frame...")
        ref_features = self.detect_features_3d(
            volumes[..., reference_frame_idx],
            reference_bbox
        )
        
        total_features = sum(len(f) for f in ref_features)
        print(f"  Detected {total_features} features across {len(ref_features)} slices")
        
        # Track forward from reference frame
        print(f"\nTracking forward...")
        current_bbox = reference_bbox
        current_features = ref_features
        
        for frame_idx in range(reference_frame_idx + 1, n_frames):
            print(f"  Frame {frame_idx}/{n_frames-1}...", end='\r')
            
            # Track features using TRUE 3D
            next_features, status, motion_vectors = self.track_features_3d(
                volumes[..., frame_idx - 1],
                volumes[..., frame_idx],
                current_features
            )
            
            # Estimate global motion
            global_motion, confidence = self.estimate_global_motion(
                motion_vectors,
                status
            )
            
            # ALWAYS apply translation (no frame rejection)
            dx, dy, dz = global_motion.astype(int)
            current_bbox = current_bbox.translate(dx, dy, dz)
            
            # Update features for next iteration
            total_features = sum(len(f) for f in next_features) if next_features else 0
            
            if confidence < self.min_features_ratio or total_features < 10:
                # Re-detect features in current bbox to avoid drift
                print(f"\n  Frame {frame_idx}: Low confidence ({confidence:.2f}) or few features ({total_features}), re-detecting...")
                current_features = self.detect_features_3d(
                    volumes[..., frame_idx],
                    current_bbox
                )
            else:
                current_features = next_features
            
            tracked_bboxes[frame_idx] = current_bbox
            confidences[frame_idx] = confidence
        
        print()  # New line after progress
        
        # Track backward from reference frame
        print(f"\nTracking backward...")
        current_bbox = reference_bbox
        current_features = ref_features
        
        for frame_idx in range(reference_frame_idx - 1, -1, -1):
            print(f"  Frame {frame_idx}...", end='\r')
            
            # Track features using TRUE 3D
            next_features, status, motion_vectors = self.track_features_3d(
                volumes[..., frame_idx + 1],
                volumes[..., frame_idx],
                current_features
            )
            
            # Estimate global motion
            global_motion, confidence = self.estimate_global_motion(
                motion_vectors,
                status
            )
            
            # ALWAYS apply translation
            dx, dy, dz = global_motion.astype(int)
            current_bbox = current_bbox.translate(dx, dy, dz)
            
            # Update features
            total_features = sum(len(f) for f in next_features) if next_features else 0
            
            if confidence < self.min_features_ratio or total_features < 10:
                print(f"\n  Frame {frame_idx}: Low confidence ({confidence:.2f}), re-detecting...")
                current_features = self.detect_features_3d(
                    volumes[..., frame_idx],
                    current_bbox
                )
            else:
                current_features = next_features
            
            tracked_bboxes[frame_idx] = current_bbox
            confidences[frame_idx] = confidence
        
        print()  # New line
        print("\nTracking complete!")
        
        return tracked_bboxes, confidences


def visualize_optical_flow_tracking(
    volumes: np.ndarray,  # (X, Y, Z, T)
    tracked_bboxes: List[BoundingBox3D],
    confidences: List[float],
    output_path: str = 'optical_flow_tracking.png'
):
    """
    Visualize optical flow tracking results
    
    Args:
        volumes: All volume frames (X, Y, Z, T)
        tracked_bboxes: List of tracked bounding boxes (all valid)
        confidences: List of confidence values
        output_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    n_frames = len(tracked_bboxes)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot confidence over time
    frames = np.arange(n_frames)
    axes[0].plot(frames, confidences, 'b-o', linewidth=2, markersize=6)
    axes[0].axhline(0.3, color='r', linestyle='--', alpha=0.5, 
                   label='Low confidence threshold')
    axes[0].set_xlabel('Frame Number', fontsize=12)
    axes[0].set_ylabel('Tracking Confidence', fontsize=12)
    axes[0].set_title('3D Optical Flow Tracking Confidence', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Plot bbox center movement (all frames valid)
    centers = np.array([bbox.center for bbox in tracked_bboxes])
    
    axes[1].plot(frames, centers[:, 0], 'r-', label='X position (lateral)', alpha=0.7, linewidth=2)
    axes[1].plot(frames, centers[:, 1], 'g-', label='Y position (depth)', alpha=0.7, linewidth=2)
    axes[1].plot(frames, centers[:, 2], 'b-', label='Z position (elevational)', alpha=0.7, linewidth=2)
    axes[1].set_xlabel('Frame Number', fontsize=12)
    axes[1].set_ylabel('Bbox Center Position (pixels)', fontsize=12)
    axes[1].set_title('3D Bounding Box Movement', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved tracking visualization to {output_path}")