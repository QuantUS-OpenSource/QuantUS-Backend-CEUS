"""
2D motion compensation for dual-display CEUS cines, ported verbatim from the
legacy QuantUS tool (Legacy-QuantUS/src/CeusMcTool2d/roiSelection_ui_helper.py
perform_MC, and src/Utils/motionCorrection.py).

Method: Thodsawit Tiyarattanachai, MD, "A Comprehensive Motion Compensation
Method for In-Plane and Out-of-Plane Motion in Dynamic Contrast-Enhanced
Ultrasound of Focal Liver Lesions" (2022).

The ROI's bounding box on the reference frame is template-matched
(cv2.TM_CCOEFF_NORMED) on the B-mode panel frame by frame. A frame whose match
falls below an adaptive correlation threshold is treated as out of plane and
gets no box, which drops it from the TIC. The threshold starts at the median
whole-cine correlation and is lowered by 0.02 per pass until a box found in the
previous pass moves (that previous pass is kept) or it drops below every
frame's correlation.

Axis convention: frames are (T, rows, cols); boxes are (x0, y0, w, h) with x a
column index and y a row index, as in the legacy code.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

Bbox = Tuple[int, int, int, int]


def find_correlation(bmode, ref_bmodes, set_quantile):
    # works for multiple reference frames
    # based on cv2.matchTemplate

    n_refs = len(ref_bmodes)
    n_frames = bmode.shape[0]
    correlations = np.zeros((n_refs, n_frames))

    for ref_idx in range(n_refs):
        ref = ref_bmodes[ref_idx]
        for frame in range(n_frames):
            similarity_map = cv2.matchTemplate(bmode[frame], ref, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(similarity_map)
            correlations[ref_idx, frame] = max_val

    correlations = np.mean(correlations, axis=0)
    threshold = np.quantile(correlations, set_quantile)

    return correlations, threshold


def compute_similarity_map(search_region, ref_patches, current_ref_idx):
    n_refs = len(ref_patches)
    corrs = np.zeros((n_refs,))

    for ref_idx in range(n_refs):
        ref = ref_patches[ref_idx]
        similarity_map = cv2.matchTemplate(search_region, ref, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(similarity_map)
        corrs[ref_idx] = max_val
        if current_ref_idx == ref_idx:
            current_max_loc = max_loc

    mean_corr = np.mean(corrs)

    return mean_corr, current_max_loc


def check_bbox_move(previous_all_lesion_bboxes, all_lesion_bboxes):
    bbox_move = False

    for frame in range(len(previous_all_lesion_bboxes)):
        previous_bbox = previous_all_lesion_bboxes[frame]
        current_bbox = all_lesion_bboxes[frame]
        if not (previous_bbox is None):
            if not (previous_bbox == current_bbox):
                bbox_move = True
                break

    return bbox_move


@dataclass
class MotionCompensation2DResult:
    """Per-frame ROI boxes from the 2D tracker. None marks a rejected frame."""
    bboxes: List[Optional[Bbox]]
    reference_frame: int
    correlations: List[Optional[float]]
    threshold: float
    search_margin: int
    kept_frames: List[int] = field(init=False)

    def __post_init__(self):
        self.kept_frames = [t for t, b in enumerate(self.bboxes) if b is not None]

    def frame_mask(self, frame: int, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Boolean ROI mask for `frame` (the tracked box), or None if rejected.

        Clipping at the bottom/right edge follows the legacy code.
        """
        bbox = self.bboxes[frame]
        if bbox is None:
            return None
        x0, y0, x_len, y_len = bbox
        if y0 + y_len >= shape[0]:
            y_len = shape[0] - y0 - 1
        if x0 + x_len >= shape[1]:
            x_len = shape[1] - x0 - 1
        mask = np.zeros(shape, dtype=bool)
        mask[y0: y0 + y_len, x0: x0 + x_len] = True
        return mask

    @property
    def translation_vectors(self) -> List[Optional[Tuple[int, int]]]:
        """Per-frame (d_row, d_col) shift of the tracked box from the reference
        frame's box, in pixels; None for rejected frames.

        Ordered like the mask axes, so component i shifts axis i of a (rows, cols)
        mask, matching the 3D translation_vectors convention.
        """
        ref_x0, ref_y0 = self.bboxes[self.reference_frame][:2]
        return [None if b is None else (int(b[1] - ref_y0), int(b[0] - ref_x0)) for b in self.bboxes]

    def to_dict(self) -> dict:
        return {
            'translation_vectors': [None if t is None else list(t) for t in self.translation_vectors],
            'bboxes': [None if b is None else [int(v) for v in b] for b in self.bboxes],
            'reference_frame': int(self.reference_frame),
            'correlations': [None if c is None else float(c) for c in self.correlations],
            'threshold': float(self.threshold),
            'search_margin': int(self.search_margin),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MotionCompensation2DResult':
        return cls(
            bboxes=[None if b is None else tuple(b) for b in d['bboxes']],
            reference_frame=d['reference_frame'],
            correlations=d['correlations'],
            threshold=d['threshold'],
            search_margin=d['search_margin'],
        )

    def to_json(self) -> str:
        return json.dumps({'motion_compensation_2d': self.to_dict()})


def track_motion_2d(bmode_gray: np.ndarray, roi_mask: np.ndarray, ref_frame: int,
                    search_margin: Optional[int] = None) -> MotionCompensation2DResult:
    """Track the ROI through the B-mode cine with the legacy 2D algorithm.

    Args:
        bmode_gray: (T, rows, cols) uint8 B-mode panel
        roi_mask: (rows, cols) ROI drawn on ref_frame; only its bounding box is
            tracked
        ref_frame: frame the ROI was drawn on
        search_margin: pixels searched beyond the previous box on each side.
            Defaults to the legacy int((0.5 / 15) * panel height).

    Returns:
        MotionCompensation2DResult with one box (or None) per frame.
    """
    # Credit Thodsawit Tiyarattanachai, MD. See module docstring for citation.
    fullGrayArray = np.ascontiguousarray(bmode_gray, dtype=np.uint8)
    h_bmode = fullGrayArray.shape[1]

    set_quantile = 0.50
    step = 1  # fullFrameRate. step=2 for halfFrameRate
    threshold_decrease_per_step = 0.02

    # get ref frame
    if search_margin is None:
        search_margin = int((0.5 / 15) * h_bmode)
    ref_frames = [ref_frame]
    mask = roi_mask
    pos_coor = np.argwhere(mask > 0)
    x_values = pos_coor[:, 1]
    y_values = pos_coor[:, 0]
    x0 = x_values.min()
    x1 = x_values.max()
    w = x1 - x0 + 1
    y0 = y_values.min()
    y1 = y_values.max()
    h = y1 - y0 + 1
    bboxes = [(int(x0), int(y0), int(w), int(h))]

    ###################################################
    # find initial correlation in the first run
    # max(0, ...) is not in the legacy code; see _search_region
    min_x0 = max(0, min([e[0] for e in bboxes]) - search_margin)
    max_x1 = max([e[0] + e[2] for e in bboxes]) + search_margin
    min_y0 = max(0, min([e[1] for e in bboxes]) - search_margin)
    max_y1 = max([e[1] + e[3] for e in bboxes])
    bmode = fullGrayArray[:, min_y0:max_y1, min_x0:max_x1]
    ref_f = ref_frames[0]
    ref_b = bboxes[0]
    ref_bmodes = [
        fullGrayArray[
            ref_f,
            ref_b[1]: ref_b[1] + ref_b[3],
            ref_b[0]: ref_b[0] + ref_b[2],
        ]
    ]

    _, threshold = find_correlation(bmode, ref_bmodes, set_quantile)
    ###################################################

    ref_patches = ref_bmodes[:]
    n_frames = fullGrayArray.shape[0]

    previous_all_lesion_bboxes = [None] * n_frames
    previous_corr_with_ref = [None] * n_frames
    previous_threshold = None

    while True:
        all_search_bboxes = [None] * n_frames
        all_lesion_bboxes = [None] * n_frames
        corr_with_ref = [None] * n_frames

        for ref_idx in range(len(ref_frames)):
            ref_frame_i = ref_frames[ref_idx]
            ref_bbox = bboxes[ref_idx]

            # There is only 1 ref_frame
            ref_begin = 0
            ref_end = n_frames

            # forward tracking
            ##############################################################
            if ref_frame_i < ref_end - 1:  # can forward track only if there are frames after the ref_frame
                previous_bbox = ref_bbox
                valid = True

                for frame in range(ref_frame_i, ref_end, step):
                    full_frame = fullGrayArray[frame]
                    search_x0, search_y0, search_w, search_h, search_region = _search_region(
                        full_frame, valid, previous_bbox, ref_bbox, search_margin,
                        all_search_bboxes, all_search_bboxes[ref_frame_i + 1: ref_end])
                    if valid:
                        all_search_bboxes[frame] = (search_x0, search_y0, search_w, search_h)

                    mean_corr, max_loc = compute_similarity_map(search_region, ref_patches, ref_idx)
                    corr_with_ref[frame] = mean_corr

                    if mean_corr >= threshold:
                        valid = True
                        current_bbox = (search_x0 + max_loc[0], search_y0 + max_loc[1],
                                        ref_bbox[2], ref_bbox[3])
                        all_lesion_bboxes[frame] = current_bbox[:]
                        previous_bbox = current_bbox[:]
                    else:
                        valid = False

            # backward tracking
            ##############################################################
            # As in the legacy code, this sits inside the forward-tracking
            # branch, so a reference on the last frame is not tracked backward.
                if ref_frame_i > ref_begin:
                    previous_bbox = ref_bbox
                    valid = True

                    for frame in range(ref_frame_i - 1, ref_begin - 1, -step):
                        full_frame = fullGrayArray[frame]
                        search_x0, search_y0, search_w, search_h, search_region = _search_region(
                            full_frame, valid, previous_bbox, ref_bbox, search_margin,
                            all_search_bboxes, all_search_bboxes[ref_begin:ref_frame_i])
                        if valid:
                            all_search_bboxes[frame] = (search_x0, search_y0, search_w, search_h)

                        mean_corr, max_loc = compute_similarity_map(search_region, ref_patches, ref_idx)
                        corr_with_ref[frame] = mean_corr

                        if mean_corr >= threshold:
                            valid = True
                            current_bbox = (search_x0 + max_loc[0], search_y0 + max_loc[1],
                                            ref_bbox[2], ref_bbox[3])
                            all_lesion_bboxes[frame] = current_bbox[:]
                            previous_bbox = current_bbox[:]
                        else:
                            valid = False

        # check if lesion bbox in any frame move in this iteration
        #####################
        bbox_move = check_bbox_move(previous_all_lesion_bboxes, all_lesion_bboxes)
        if bbox_move or (threshold < min([e for e in corr_with_ref if not (e is None)])):
            break
        #####################

        previous_all_lesion_bboxes = all_lesion_bboxes[:]
        previous_corr_with_ref = corr_with_ref[:]
        previous_threshold = threshold
        threshold -= threshold_decrease_per_step

    if previous_threshold is None:
        # The legacy tool reached this with no completed pass and printed the
        # same message, leaving every frame without a box.
        raise ValueError("MC not possible. Must choose a better ROI")

    return MotionCompensation2DResult(
        bboxes=previous_all_lesion_bboxes,
        reference_frame=ref_frame,
        correlations=previous_corr_with_ref,
        threshold=previous_threshold,
        search_margin=search_margin,
    )


def _search_region(full_frame, valid, previous_bbox, ref_bbox, search_margin,
                   all_search_bboxes, prior_search_bboxes):
    """Search window for one frame, as in the legacy loop.

    While tracking is valid the window is the previous box grown by
    search_margin. After a rejected frame it widens to cover the outlier-trimmed
    (1.5 IQR) range of earlier search windows.
    """
    if valid:
        search_w = int(previous_bbox[2] + (2 * search_margin))
        search_h = int(previous_bbox[3] + (2 * search_margin))
        search_x0 = int(previous_bbox[0] - ((search_w - previous_bbox[2]) / 2))
        search_y0 = int(previous_bbox[1] - ((search_h - previous_bbox[3]) / 2))
    else:
        all_search_x0 = [b[0] for b in prior_search_bboxes if not (b is None)]
        median_x0 = np.median(all_search_x0)
        IQR_x0 = np.quantile(all_search_x0, 0.75) - np.quantile(all_search_x0, 0.25)
        all_search_x0 = [x for x in all_search_x0
                         if (x >= median_x0 - (1.5 * IQR_x0)) and (x <= median_x0 + (1.5 * IQR_x0))]
        min_search_x0 = min(all_search_x0)
        max_search_x0 = max(all_search_x0)

        all_search_y0 = [b[1] for b in prior_search_bboxes if not (b is None)]
        median_y0 = np.median(all_search_y0)
        IQR_y0 = np.quantile(all_search_y0, 0.75) - np.quantile(all_search_y0, 0.25)
        all_search_y0 = [y for y in all_search_y0
                         if (y >= median_y0 - (1.5 * IQR_y0)) and (y <= median_y0 + (1.5 * IQR_y0))]
        min_search_y0 = min(all_search_y0)
        max_search_y0 = max(all_search_y0)

        search_x0 = min_search_x0
        search_y0 = min_search_y0
        search_w = (max_search_x0 - min_search_x0) + int(ref_bbox[2] + (2 * search_margin))
        search_h = (max_search_y0 - min_search_y0) + int(previous_bbox[3] + (2 * search_margin))

    # Not in the legacy code: a window starting above or left of the frame
    # makes the slice wrap to an empty region and cv2.matchTemplate abort, so
    # clip the start to 0 and keep the far edge. Windows inside the frame are
    # unchanged, so results match the legacy tool wherever it ran.
    if search_x0 < 0:
        search_w += search_x0
        search_x0 = 0
    if search_y0 < 0:
        search_h += search_y0
        search_y0 = 0

    search_region = full_frame[search_y0: search_y0 + search_h,
                               search_x0: search_x0 + search_w]
    return search_x0, search_y0, search_w, search_h, search_region
