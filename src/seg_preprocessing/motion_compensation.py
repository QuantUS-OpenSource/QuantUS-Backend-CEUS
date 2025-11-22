"""
3D Motion Compensation for CEUS Analysis
This module implements 3D motion compensation for contrast-enhanced ultrasound (CEUS) imaging
using B-mode reference data. It can be integrated into the QuantUS-plugins-CEUS workflow
as a segmentation preprocessing step.

The approach:
1. Extract bounding box from segmentation mask
2. Track bounding box center across frames using B-mode correlation
3. Apply motion compensation shifts to both CEUS and segmentation data
"""

import numpy as np
from scipy.ndimage import shift as scipy_shift
from scipy.signal import correlate
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .decorators import required_kwargs
from ..data_objs.image import UltrasoundImage
from ..data_objs.seg import CeusSeg


@dataclass
class BoundingBox3D:
    """
    3D Bounding Box class to encapsulate a region of interest.

    Attributes:
        z_min, z_max: Z-axis bounds (depth/elevation)
        y_min, y_max: Y-axis bounds (axial)
        x_min, x_max: X-axis bounds (lateral)
        center: Center coordinates (z, y, x)
    """
    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int

    def __post_init__(self):
        """Calculate center after initialization"""
        self.center = self.get_center()

    def get_center(self) -> Tuple[float, float, float]:
        """Calculate the center of the bounding box"""
        z_center = (self.z_min + self.z_max) / 2.0
        y_center = (self.y_min + self.y_max) / 2.0
        x_center = (self.x_min + self.x_max) / 2.0
        return (z_center, y_center, x_center)

    def get_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the bounding box"""
        return (
            self.z_max - self.z_min,
            self.y_max - self.y_min,
            self.x_max - self.x_min
        )

    def extract_roi(self, volume: np.ndarray) -> np.ndarray:
        """
        Extract the region defined by this bounding box from a volume.

        Args:
            volume: 3D or 4D array (Z, Y, X) or (T, Z, Y, X)

        Returns:
            Extracted ROI
        """
        if volume.ndim == 3:
            return volume[self.z_min:self.z_max,
                         self.y_min:self.y_max,
                         self.x_min:self.x_max].copy()
        elif volume.ndim == 4:
            return volume[:, self.z_min:self.z_max,
                         self.y_min:self.y_max,
                         self.x_min:self.x_max].copy()
        else:
            raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")

    def shift(self, dz: float, dy: float, dx: float) -> 'BoundingBox3D':
        """
        Create a new BoundingBox3D shifted by the given amounts.

        Args:
            dz, dy, dx: Shift amounts in each dimension

        Returns:
            New shifted BoundingBox3D
        """
        return BoundingBox3D(
            z_min=int(np.round(self.z_min + dz)),
            z_max=int(np.round(self.z_max + dz)),
            y_min=int(np.round(self.y_min + dy)),
            y_max=int(np.round(self.y_max + dy)),
            x_min=int(np.round(self.x_min + dx)),
            x_max=int(np.round(self.x_max + dx))
        )

    @classmethod
    def from_mask(cls, mask: np.ndarray, padding: int = 0) -> 'BoundingBox3D':
        """
        Create a bounding box from a binary segmentation mask.
        The bounding box will be the smallest rectangular box that encloses all
        non-zero voxels in the mask.

        Args:
            mask: Binary 3D segmentation mask
            padding: Additional padding to add around the mask (in voxels)

        Returns:
            BoundingBox3D object
        """
        if mask.ndim != 3:
            raise ValueError(f"Mask must be 3D, got shape {mask.shape}")

        # Find non-zero voxels
        nonzero_coords = np.argwhere(mask > 0)

        if len(nonzero_coords) == 0:
            raise ValueError("Mask contains no non-zero voxels")

        # Get min and max along each axis
        z_min = max(0, nonzero_coords[:, 0].min() - padding)
        z_max = min(mask.shape[0], nonzero_coords[:, 0].max() + 1 + padding)
        y_min = max(0, nonzero_coords[:, 1].min() - padding)
        y_max = min(mask.shape[1], nonzero_coords[:, 1].max() + 1 + padding)
        x_min = max(0, nonzero_coords[:, 2].min() - padding)
        x_max = min(mask.shape[2], nonzero_coords[:, 2].max() + 1 + padding)

        return cls(z_min, z_max, y_min, y_max, x_min, x_max)


class MotionCompensation3D:
    """
    3D Motion Compensation using normalized cross-correlation.

    This class implements motion tracking and compensation for 3D+time ultrasound data.
    It uses B-mode reference images to track motion and applies compensation to both
    the CEUS data and the segmentation mask.
    """

    def __init__(
        self,
        bmode_volume: np.ndarray,
        reference_frame: int = 0,
        search_range: Tuple[int, int, int] = (5, 5, 5),
        correlation_threshold: float = 0.3,
        use_reference_bbox: bool = True
    ):
        """
        Initialize the motion compensation object.

        Args:
            bmode_volume: 4D B-mode volume (T, Z, Y, X)
            reference_frame: Frame index to use as reference
            search_range: (z, y, x) search range in pixels for correlation
            correlation_threshold: Minimum correlation to accept a shift
            use_reference_bbox: If True, always correlate with reference frame.
                              If False, correlate with previous frame.
        """
        if bmode_volume.ndim != 4:
            raise ValueError(f"B-mode volume must be 4D (T, Z, Y, X), got shape {bmode_volume.shape}")

        self.bmode_volume = bmode_volume.astype(np.float32)
        self.n_frames = bmode_volume.shape[0]
        self.reference_frame = reference_frame
        self.search_range = search_range
        self.correlation_threshold = correlation_threshold
        self.use_reference_bbox = use_reference_bbox

        # Storage for computed shifts
        self.shifts: Optional[np.ndarray] = None  # Shape: (T, 3)
        self.correlation_scores: Optional[np.ndarray] = None  # Shape: (T,)

    def compute_shifts(self, reference_bbox: BoundingBox3D, verbose: bool = True) -> np.ndarray:
        """
        Compute motion shifts for all frames relative to reference frame.

        Uses vectorized normalized cross-correlation to efficiently compute shifts.

        Args:
            reference_bbox: Bounding box defining the tracking region
            verbose: Whether to print progress information

        Returns:
            Array of shifts with shape (T, 3) where each row is (dz, dy, dx)
        """
        # Extract reference ROI
        ref_roi = reference_bbox.extract_roi(self.bmode_volume[self.reference_frame])
        ref_shape = ref_roi.shape

        # Normalize reference ROI
        ref_roi_norm = self._normalize_volume(ref_roi)

        # Initialize storage
        self.shifts = np.zeros((self.n_frames, 3), dtype=np.float32)
        self.correlation_scores = np.zeros(self.n_frames, dtype=np.float32)

        # Reference frame has zero shift and perfect correlation
        self.correlation_scores[self.reference_frame] = 1.0

        # Build search boundary for all frames at once
        # This is the key vectorization insight from the author's notes
        search_boundary = self._build_search_boundary(reference_bbox)

        if verbose:
            print(f"Computing motion shifts for {self.n_frames} frames...")
            print(f"Reference ROI shape: {ref_shape}")
            print(f"Search boundary shape: {search_boundary.shape}")

        # Compute correlation map using vectorized convolution
        # This implements the author's idea: "convolve it with the reference VOI"
        correlation_map = self._compute_correlation_map_vectorized(
            ref_roi_norm, search_boundary
        )

        if verbose:
            print(f"Correlation map shape: {correlation_map.shape}")

        # Find best shifts for each frame
        for t in range(self.n_frames):
            if t == self.reference_frame:
                continue

            # Find maximum correlation in this frame's correlation map
            corr_frame = correlation_map[t]
            max_idx = np.unravel_index(np.argmax(corr_frame), corr_frame.shape)
            max_corr = corr_frame[max_idx]

            # Convert index to shift (relative to center of search range)
            dz = max_idx[0] - self.search_range[0]
            dy = max_idx[1] - self.search_range[1]
            dx = max_idx[2] - self.search_range[2]

            self.shifts[t] = [dz, dy, dx]
            self.correlation_scores[t] = max_corr

            if verbose and (t % 10 == 0 or t == self.n_frames - 1):
                print(f"Frame {t}/{self.n_frames-1}: shift=({dz:.1f}, {dy:.1f}, {dx:.1f}), corr={max_corr:.3f}")

        # Apply threshold - frames below threshold get zero shift
        below_threshold = self.correlation_scores < self.correlation_threshold
        if np.any(below_threshold):
            n_below = np.sum(below_threshold)
            if verbose:
                print(f"\nWarning: {n_below} frames below correlation threshold {self.correlation_threshold}")
                print(f"These frames will not be motion compensated.")
            # Keep shifts at zero for these frames
            self.shifts[below_threshold] = 0

        return self.shifts

    def _build_search_boundary(self, bbox: BoundingBox3D) -> np.ndarray:
        """
        Build a search boundary volume for all frames.

        This implements the vectorization idea: create a larger volume that includes
        the search range, then we can apply correlation as a convolution operation.

        Args:
            bbox: Reference bounding box

        Returns:
            Array of shape (T, Z+2*sz, Y+2*sy, X+2*sx) where s* are search ranges
        """
        sz, sy, sx = self.search_range

        # Expanded bounding box
        z_min = max(0, bbox.z_min - sz)
        z_max = min(self.bmode_volume.shape[1], bbox.z_max + sz)
        y_min = max(0, bbox.y_min - sy)
        y_max = min(self.bmode_volume.shape[2], bbox.y_max + sy)
        x_min = max(0, bbox.x_min - sx)
        x_max = min(self.bmode_volume.shape[3], bbox.x_max + sx)

        # Extract expanded region for all frames
        search_boundary = self.bmode_volume[:, z_min:z_max, y_min:y_max, x_min:x_max].copy()

        # Normalize each frame
        for t in range(self.n_frames):
            search_boundary[t] = self._normalize_volume(search_boundary[t])

        return search_boundary

    def _compute_correlation_map_vectorized(
        self,
        reference_roi: np.ndarray,
        search_boundary: np.ndarray
    ) -> np.ndarray:
        """
        Compute normalized cross-correlation map using vectorized operations.

        This is the core vectorization: we compute correlations for all possible
        shifts simultaneously by sliding the reference ROI over the search boundary.

        Args:
            reference_roi: Normalized reference ROI (Z, Y, X)
            search_boundary: Normalized search region for all frames (T, Z+2*sz, Y+2*sy, X+2*sx)

        Returns:
            Correlation map of shape (T, 2*sz+1, 2*sy+1, 2*sx+1)
        """
        n_frames = search_boundary.shape[0]
        ref_shape = reference_roi.shape
        sz, sy, sx = self.search_range

        # Output correlation map
        corr_map = np.zeros((n_frames, 2*sz+1, 2*sy+1, 2*sx+1), dtype=np.float32)

        # Slide the reference ROI over the search boundary
        for dz in range(2*sz + 1):
            for dy in range(2*sy + 1):
                for dx in range(2*sx + 1):
                    # Extract corresponding region from search boundary
                    z_start, z_end = dz, dz + ref_shape[0]
                    y_start, y_end = dy, dy + ref_shape[1]
                    x_start, x_end = dx, dx + ref_shape[2]

                    search_roi = search_boundary[:, z_start:z_end, y_start:y_end, x_start:x_end]

                    # Compute normalized correlation for all frames at once
                    # This is just the dot product since both are already normalized
                    correlation = np.sum(reference_roi[None, ...] * search_roi, axis=(1, 2, 3))
                    correlation /= np.prod(ref_shape)  # Normalize by number of voxels

                    corr_map[:, dz, dy, dx] = correlation

        return corr_map

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize a volume to zero mean and unit standard deviation.

        Args:
            volume: Input volume

        Returns:
            Normalized volume
        """
        volume = volume.astype(np.float32)
        mean = np.mean(volume)
        std = np.std(volume)

        if std < 1e-6:  # Avoid division by zero
            return np.zeros_like(volume)

        return (volume - mean) / std

    def apply_compensation(
        self,
        ceus_volume: np.ndarray,
        seg_mask: Optional[np.ndarray] = None,
        order: int = 1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply computed motion compensation shifts to CEUS data and segmentation mask.

        Args:
            ceus_volume: 4D CEUS volume (T, Z, Y, X) or 3D (Z, Y, X)
            seg_mask: Optional 3D segmentation mask to also shift
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)

        Returns:
            Tuple of (compensated_ceus, compensated_seg_mask)
        """
        if self.shifts is None:
            raise ValueError("Must call compute_shifts() before apply_compensation()")

        is_4d = ceus_volume.ndim == 4

        if not is_4d and ceus_volume.ndim != 3:
            raise ValueError(f"CEUS volume must be 3D or 4D, got shape {ceus_volume.shape}")

        # If 3D, add time dimension
        if not is_4d:
            ceus_volume = ceus_volume[np.newaxis, ...]

        if ceus_volume.shape[0] != self.n_frames:
            raise ValueError(f"CEUS volume has {ceus_volume.shape[0]} frames but "
                           f"motion compensation computed for {self.n_frames} frames")

        # Apply shifts to CEUS volume
        compensated_ceus = np.zeros_like(ceus_volume)
        for t in range(self.n_frames):
            shift_vector = -self.shifts[t]  # Negative because we're shifting image back
            compensated_ceus[t] = scipy_shift(
                ceus_volume[t],
                shift_vector,
                order=order,
                mode='nearest'
            )

        # Apply shifts to segmentation mask if provided
        compensated_seg = None
        if seg_mask is not None:
            if seg_mask.ndim != 3:
                raise ValueError(f"Segmentation mask must be 3D, got shape {seg_mask.shape}")

            # Use the reference frame shift (typically zero)
            ref_shift = -self.shifts[self.reference_frame]
            compensated_seg = scipy_shift(
                seg_mask,
                ref_shift,
                order=0,  # Nearest neighbor for binary mask
                mode='nearest'
            ).astype(seg_mask.dtype)

        # Remove time dimension if input was 3D
        if not is_4d:
            compensated_ceus = compensated_ceus[0]

        return compensated_ceus, compensated_seg

    def get_quality_metrics(self) -> dict:
        """
        Get quality metrics for the motion compensation.

        Returns:
            Dictionary with quality metrics
        """
        if self.shifts is None:
            raise ValueError("Must call compute_shifts() first")

        return {
            'mean_correlation': np.mean(self.correlation_scores),
            'min_correlation': np.min(self.correlation_scores),
            'max_shift_z': np.max(np.abs(self.shifts[:, 0])),
            'max_shift_y': np.max(np.abs(self.shifts[:, 1])),
            'max_shift_x': np.max(np.abs(self.shifts[:, 2])),
            'mean_shift_magnitude': np.mean(np.linalg.norm(self.shifts, axis=1)),
            'frames_below_threshold': np.sum(self.correlation_scores < self.correlation_threshold)
        }
