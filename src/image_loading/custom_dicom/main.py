import os
import cv2
import pydicom 
import numpy as np

from ...data_objs.image import UltrasoundImage

class EntryClass(UltrasoundImage):
    """
    Loader class for DICOM CEUS image data.

    This class parses CEUS data from DICOM files, extracting pixel data, pixel dimensions,
    frame rate for the scan. The input is a folder containing DICOM files, and the loader identifies
    the largest DICOM file (by size) to extract the relevant metadata and pixel data for that file only.
    The following attributes are set:
        - pixel_data, pixdim, frame_rate: for the scan
    Output pixel data is in uint8 format with (sagittal, coronal, axial, time) dimensions.

    Kwargs:
        - transpose: whether to transpose the pixel data (default False).
    """
    required_kwargs = []
    extensions = ["FOLDER"]
    spatial_dims = 2
    
    def __init__(self, scan_path: str, **kwargs):
        super().__init__(scan_path)
        
        # Supported file extensions for this loader
        assert os.path.isdir(scan_path), "Input path must be a folder!"

        dicom_sizes = {}
        for fname in os.listdir(scan_path):
            if fname.startswith('A'):
                file_size = os.path.getsize(os.path.join(scan_path, fname))
                dicom_sizes[os.path.join(scan_path, fname)] = file_size
        
        max_key = max(dicom_sizes, key=dicom_sizes.get)
        vid = pydicom.dcmread(max_key)
        pixel_size_x = vid.SequenceOfUltrasoundRegions[0].PhysicalDeltaX  # in mm
        pixel_size_y = vid.SequenceOfUltrasoundRegions[0].PhysicalDeltaY  # in mm
        self.pixdim = [pixel_size_y, pixel_size_x]
        self.frame_rate = float(vid.FrameTime) / 1000  # convert ms to s

        self.pixel_data = np.array(vid.pixel_array)
        self.intensities_for_analysis = np.array(
            [cv2.cvtColor(self.pixel_data[i], cv2.COLOR_BGR2GRAY) for i in range(self.pixel_data.shape[0])]
        )
