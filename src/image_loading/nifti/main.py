import numpy as np
import nibabel as nib

from ...data_objs.image import UltrasoundImage

class EntryClass(UltrasoundImage):
    """
    Loader class for NIfTI CEUS image data.

    This class parses CEUS data from NIfTI files, extracting pixel data, pixel dimensions,
    frame rate for the scan.
    The following attributes are set:
        - pixel_data, pixdim, frame_rate: for the scan
    Output pixel data is in uint8 format with (sagittal, coronal, axial, time) dimensions.

    Kwargs:
        - transpose: whether to transpose the pixel data (default False).
    """
    required_kwargs = ['transpose']
    extensions = [".nii", ".nii.gz"]
    spatial_dims = 3
    
    def __init__(self, scan_path: str, **kwargs):
        super().__init__(scan_path)
        
        # Supported file extensions for this loader
        assert max([scan_path.endswith(x) for x in self.extensions]), f"File must end with {self.extensions}"
        
        img = nib.load(scan_path)
        header = img.header
        pixdim = header.get_zooms()[:3]  # tuple of pixel dimensions
        frame_rate = 1.0 / header.get_zooms()[3] if len(header.get_zooms()) > 3 and header.get_zooms()[3] > 0 else 1 # s

        if kwargs.get('transpose', False):
            self.pixel_data = np.asarray(img.dataobj, dtype=np.uint8).T
        else:
            self.pixel_data = np.asarray(img.dataobj, dtype=np.uint8)

        self.pixdim = pixdim
        self.frame_rate = frame_rate
        self.intensities_for_analysis = self.pixel_data
