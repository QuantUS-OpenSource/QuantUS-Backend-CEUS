import cv2
import numpy as np
import nibabel as nib

from ...data_objs.image import UltrasoundImage

class EntryClass(UltrasoundImage):
    """
    Loader class for AVI CEUS image data.

    This class parses CEUS data from AVI files, extracting pixel data, pixel dimensions,
    frame rate for the scan.
    The following attributes are set:
        - pixel_data, pixdim, frame_rate: for the scan
    Output pixel data is in uint8 format with (sagittal, coronal, axial, time) dimensions.

    Kwargs:
        - transpose: whether to transpose the pixel data (default False).
    """
    required_kwargs = ['pix_height_mm', 'pix_width_mm']
    extensions = [".avi"]
    spatial_dims = 2
    
    def __init__(self, scan_path: str, **kwargs):
        super().__init__(scan_path)
        
        # Supported file extensions for this loader
        assert max([scan_path.endswith(x) for x in self.extensions]), f"File must end with {self.extensions}"
        
        cap = cv2.VideoCapture(scan_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = 1/cap.get(cv2.CAP_PROP_FPS)
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("No data in video file!")
            return

        # Extract RGB frames from AVI
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        pixel_data = np.zeros(
            (n_frames, first_frame.shape[0], first_frame.shape[1], 3)
        ).astype(first_frame.dtype)
        pixel_data[0] = first_frame
        for i in range(1, n_frames):
            ret, frame = cap.read()
            if not ret:
                print("Video data ended prematurely!")
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            pixel_data[i] = frame

        # Extract grayscale frames from RGB
        grayscale_arr = np.array(
            [
                cv2.cvtColor(pixel_data[i], cv2.COLOR_BGR2GRAY)
                for i in range(pixel_data.shape[0])
            ]
        )

        # Pixdims not readable from AVI file alone - must be provided
        pix_height_mm = kwargs.get('pix_height_mm', 1.0)
        pix_width_mm = kwargs.get('pix_width_mm', 1.0)

        self.pixdim = [pix_height_mm, pix_width_mm]
        self.frame_rate = frame_rate # in seconds
        self.intensities_for_analysis = grayscale_arr
        self.pixel_data = pixel_data
