from pathlib import Path

import numpy as np

from ...data_objs.image import UltrasoundImage
from .parser import parse_nlc_xml, read_nlc_raw

class EntryClass(UltrasoundImage):
    """
    Loader class for Vevo F2 Non-Linear Contrast Mode raw CEUS data
    (FUJIFILM VisualSonics Vevo F2, ".raw.nlc" + companion ".raw.xml").

    Ported from VsiNlcModeRawRead.m. Assumes a "<basename>.raw.xml" parameter
    file sits alongside "<basename>.raw.nlc".
    The following attributes are set:
        - pixel_data, pixdim, frame_rate: for the scan
    intensities_for_analysis holds the raw *linear* amplitude data (not log-compressed);
    pixel_data holds a log-compressed uint8 rendering for display, both with
    (time, depth/axial, width/lateral) dimensions.

    Kwargs:
        - frame_indices: 1-based list of frame numbers to load (default: all frames).
          The raw format supports true random-access reads, so this can be used to
          avoid loading an entire large acquisition into memory at once.
        - dynamic_range_db: dB range below the display ceiling used when log-compressing
          to build the uint8 pixel_data rendering (default 50.0). Does not affect
          intensities_for_analysis.
        - display_ceiling_percentile: percentile of the (pooled, all-frames) linear
          amplitude data used as the 0 dB reference for the uint8 rendering (default
          99.5). NLC data commonly has a handful of much-brighter-than-everything-else
          pixels; referencing the true max crushes the rest of the contrast signal to
          near-black, so a high percentile is used instead. Does not affect
          intensities_for_analysis.
        - transpose: whether to transpose the (depth, width) axes (default False).
    """
    required_kwargs = []
    extensions = [".raw.nlc"]
    spatial_dims = 2

    def __init__(self, scan_path: str, **kwargs):
        super().__init__(scan_path)

        assert scan_path.endswith(self.extensions[0]), f"File must end with {self.extensions}"

        self.scan_name = Path(scan_path).name[:-len(self.extensions[0])]  # strip '.raw.nlc'

        xml_path = scan_path[:-len(".nlc")] + ".xml"
        assert Path(xml_path).is_file(), \
            f"Companion parameter file not found (expected '{xml_path}')"

        params = parse_nlc_xml(xml_path)
        num_samples, num_lines = params["num_samples"], params["num_lines"]

        frame_indices = kwargs.get("frame_indices", None)
        data, timestamps_ms = read_nlc_raw(scan_path, params, frame_indices=frame_indices)

        if kwargs.get("transpose", False):
            data = np.transpose(data, (0, 2, 1))
            num_samples, num_lines = num_lines, num_samples

        if len(timestamps_ms) > 1:
            frame_rate = float(np.mean(np.diff(timestamps_ms)) / 1000.0)  # s/frame
        else:
            frame_rate = 1.0
            print("Only one frame loaded; cannot infer frame rate from timestamps. "
                  "Defaulting frame_rate to 1.0 s/frame.")

        axial_pix_mm = (params["depth_mm"] - params["depth_offset_mm"]) / max(num_samples - 1, 1)
        lateral_pix_mm = params["width_mm"] / max(num_lines - 1, 1)

        dynamic_range_db = kwargs.get("dynamic_range_db", 50.0)
        display_ceiling_percentile = kwargs.get("display_ceiling_percentile", 99.5)
        # A robust (percentile) ceiling, not the true max: a handful of outlier-bright
        # pixels would otherwise dominate the 0 dB reference and crush the rest of the
        # (real) contrast signal to near-black.
        ceiling = float(np.percentile(data, display_ceiling_percentile)) if data.size else 0.0
        floor = ceiling / (10 ** (dynamic_range_db / 20.0)) if ceiling > 0 else 1.0
        with np.errstate(divide="ignore"):
            db = 20.0 * np.log10(np.clip(data, floor, None) / floor)
        pixel_data = np.clip(db / dynamic_range_db * 255.0, 0, 255).astype(np.uint8)

        self.pixdim = [axial_pix_mm, lateral_pix_mm]
        self.frame_rate = frame_rate
        self.intensities_for_analysis = data
        self.pixel_data = pixel_data
        self.extras_dict = {
            "timestamps_ms": timestamps_ms,
            "frame_indices_loaded": frame_indices if frame_indices is not None
                                     else list(range(1, data.shape[0] + 1)),
            "depth_mm": params["depth_mm"],
            "width_mm": params["width_mm"],
            "depth_offset_mm": params["depth_offset_mm"],
            "centre_mm": params["centre_mm"],
        }
