import os
import cv2
import pydicom 
import numpy as np

from ...data_objs.image import UltrasoundImage

class EntryClass(UltrasoundImage):
    """
    Loader class for DICOM CEUS image data.

    This class parses CEUS data from a DICOM file, extracting pixel data, pixel dimensions,
    frame rate for the scan. The input is a single DICOM file.
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
        assert os.path.isdir(scan_path), "Input path must be a folder!"

        vid = pydicom.dcmread(scan_path)
        
        # Handle missing SequenceOfUltrasoundRegions
        ultrasound_regions = getattr(vid, 'SequenceOfUltrasoundRegions', None)
        if ultrasound_regions is not None and len(ultrasound_regions) > 0:
            pixel_size_x = vid.SequenceOfUltrasoundRegions[0].PhysicalDeltaX  # in mm
            pixel_size_y = vid.SequenceOfUltrasoundRegions[0].PhysicalDeltaY  # in mm
        elif hasattr(vid, 'PixelSpacing'):
            pixel_size_y, pixel_size_x = vid.PixelSpacing  # PixelSpacing is [y, x]
        else:
            # Fallback to 1.0 if no spacing info is found
            pixel_size_x = 1.0
            pixel_size_y = 1.0

        self.pixdim = [float(pixel_size_y), float(pixel_size_x)]
        
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
                        # For YBR photometric interpretations, pydicom.pixel_array typically
                        # returns data already converted to RGB. In that case we can safely 
                        # treat the frame as RGB and convert it to grayscale.
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    else:
                        if str(photometric_interpretation).upper().startswith('RGB'):
                            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        else:
                            # If photometric interpretation is not RGB or YBR, we can attempt to convert
                            # using OpenCV, but this may not always be correct. In that case, we can
                            # simply take the first channel as a fallback.
                            try:
                                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            except cv2.error:
                                gray_frame = frame[:, :, 0]  # Fallback to first channel
                    self.intensities_for_analysis.append(gray_frame)
                self.intensities_for_analysis = np.array(self.intensities_for_analysis)
        else:
            raise ValueError(f"Unexpected pixel data shape: {self.pixel_data.shape}")