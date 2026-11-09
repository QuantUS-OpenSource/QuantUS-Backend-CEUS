import os
import cv2
import pydicom 
import numpy as np

from ...data_objs.image import UltrasoundImage

class EntryClass(UltrasoundImage):
    """
    Loader class for DICOM CEUS image data.

    This class parses CEUS data from a DICOM file, extracting pixel data, pixel dimensions,
    and frame rate for the scan. The input is a single DICOM file.
    The following attributes are set:
        - pixel_data, pixdim, frame_rate: for the scan
    Output pixel data is in uint8 format with (sagittal, coronal, axial, time) dimensions.

    Kwargs:
        - transpose: whether to transpose the pixel data (default False).
    """
    required_kwargs = []
    extensions = ["dcm", "DICOM"]
    spatial_dims = 2
    
    def __init__(self, scan_path: str, **kwargs):
        super().__init__(scan_path)
        
        # Supported file extensions for this loader
        assert os.path.isfile(scan_path), f"Input path must be a file! Got: {scan_path}"

        vid = pydicom.dcmread(scan_path)
        
        # Handle missing SequenceOfUltrasoundRegions
        if hasattr(vid, 'SequenceOfUltrasoundRegions'):
            pixel_size_y, pixel_size_x = _region_pixdim_mm(vid.SequenceOfUltrasoundRegions[0])
        elif hasattr(vid, 'PixelSpacing'):
            pixel_size_y, pixel_size_x = vid.PixelSpacing  # PixelSpacing is [y, x]
        else:
            # Fallback to 1.0 if no spacing info is found
            pixel_size_x = 1.0
            pixel_size_y = 1.0
            
        self.pixdim = [pixel_size_y, pixel_size_x]

        # Handle missing FrameTime
        if hasattr(vid, 'FrameTime'):
            self.frame_rate = float(vid.FrameTime) / 1000  # convert ms to s
        elif hasattr(vid, 'CineRate'):
            self.frame_rate = 1.0 / float(vid.CineRate)
        else:
            # Fallback to a common frame rate (e.g., 30 fps)
            self.frame_rate = 1.0 / 30.0

        self.pixel_data = np.array(vid.pixel_array)

        # Handle different color spaces and channel counts
        if len(self.pixel_data.shape) == 3:  # (frames, height, width) - already grayscale
            self.intensities_for_analysis = self.pixel_data.copy()
        elif len(self.pixel_data.shape) == 4:  # (frames, height, width, channels)
            channels = self.pixel_data.shape[3]
            if channels == 1:
                self.intensities_for_analysis = self.pixel_data[:, :, :, 0]
            else:
                # Handle YBR color space if present (common in DICOM)
                photometric_interpretation = getattr(vid, 'PhotometricInterpretation', 'RGB')
                
                self.intensities_for_analysis = []
                for i in range(self.pixel_data.shape[0]):
                    frame = self.pixel_data[i]
                    if 'YBR' in photometric_interpretation:
                        # Simple YBR to Gray approximation if needed, 
                        # but often OpenCV BGR2GRAY works if pydicom already converted to RGB
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    else:
                        # Try BGR first (standard for OpenCV), then RGB if that fails or looks wrong
                        try:
                            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        except:
                            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    self.intensities_for_analysis.append(gray_frame)
                self.intensities_for_analysis = np.array(self.intensities_for_analysis)
        else:
            raise ValueError(f"Unexpected pixel data shape: {self.pixel_data.shape}")

        # Dual display: contrast and B-mode panels side by side in each frame.
        # The contrast panel becomes this image and the B-mode panel becomes
        # self.bmode_image, both cropped to the same size so a pixel in one is
        # the same tissue location in the other.
        regions = [r for r in getattr(vid, 'SequenceOfUltrasoundRegions', [])
                   if hasattr(r, 'RegionLocationMinX0')]
        if len(regions) >= 2 and self.pixel_data.ndim == 4:
            self._split_dual_display(regions[:2], scan_path)

    def _split_dual_display(self, regions, scan_path: str) -> None:
        boxes = [(int(r.RegionLocationMinX0), int(r.RegionLocationMinY0),
                  int(r.RegionLocationMaxX1), int(r.RegionLocationMaxY1)) for r in regions]
        h = min(y1 - y0 for _, y0, _, y1 in boxes)
        w = min(x1 - x0 for x0, _, x1, _ in boxes)

        # B-mode is drawn in grayscale and contrast in a colour map, so the
        # panel with less spread between colour channels is the B-mode one.
        sample = self.pixel_data[::max(1, len(self.pixel_data) // 10)].astype(np.int16)
        saturation = [np.mean(np.ptp(sample[:, y0:y0 + h, x0:x0 + w], axis=-1))
                      for x0, y0, _, _ in boxes]
        bmode_ix = int(np.argmin(saturation))
        ce_ix = 1 - bmode_ix

        def crop(arr, ix):
            x0, y0 = boxes[ix][:2]
            return np.ascontiguousarray(arr[:, y0:y0 + h, x0:x0 + w])

        bmode = UltrasoundImage(scan_path)
        bmode.pixel_data = crop(self.pixel_data, bmode_ix)
        bmode.intensities_for_analysis = crop(self.intensities_for_analysis, bmode_ix)
        bmode.pixdim = list(_region_pixdim_mm(regions[bmode_ix]))
        bmode.frame_rate = self.frame_rate
        bmode.spatial_dims = 2

        self.pixel_data = crop(self.pixel_data, ce_ix)
        self.intensities_for_analysis = crop(self.intensities_for_analysis, ce_ix)
        self.pixdim = list(_region_pixdim_mm(regions[ce_ix]))
        self.bmode_image = bmode


def _region_pixdim_mm(region):
    """(row, col) pixel size in mm of an ultrasound region.

    PhysicalDelta is stored in the region's physical units; code 3 is cm.
    """
    dy, dx = float(region.PhysicalDeltaY), float(region.PhysicalDeltaX)
    if int(getattr(region, 'PhysicalUnitsXDirection', 3)) == 3:
        dy, dx = dy * 10, dx * 10
    return dy, dx
