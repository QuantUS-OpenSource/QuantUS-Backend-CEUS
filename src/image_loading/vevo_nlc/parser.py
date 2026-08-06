import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Byte layout of a Vevo F2 ".raw.nlc" file (version >= 10):
#   [ file_header: FILE_HEADER_BYTES ]
#   for each frame:
#     [ frame_header: FRAME_HEADER_BYTES, timestamp (float64, ms) at offset +4 ]
#     [ frame data: num_samples * num_lines float32, column-major (line-major blocks) ]
FILE_HEADER_BYTES = 40
FRAME_HEADER_BYTES = 56
TIMESTAMP_OFFSET_IN_FRAME_HEADER = 4
RF_FORMAT_BIT = 0x4

NLC_XML_PARAMS = {
    "Nonlinear-Contrast-Mode/Samples": "num_samples",
    "Nonlinear-Contrast-Mode/Lines": "num_lines",
    "Nonlinear-Contrast-Mode/Depth-Offset": "depth_offset_mm",
    "Nonlinear-Contrast-Mode/Centre": "centre_mm",
    "Nonlinear-Contrast-Mode/Depth": "depth_mm",
    "Nonlinear-Contrast-Mode/Width": "width_mm",
}


def parse_nlc_xml(xml_path: str) -> dict:
    """Parse the Nonlinear-Contrast-Mode parameters out of a Vevo F2 '.raw.xml' file."""
    assert Path(xml_path).is_file(), f"XML parameter file not found: {xml_path}"

    root = ET.parse(xml_path).getroot()
    params = {}
    for node in root.iter("parameter"):
        name = node.get("name")
        if name in NLC_XML_PARAMS:
            params[NLC_XML_PARAMS[name]] = float(node.get("value").replace(",", "."))

    missing = set(NLC_XML_PARAMS.values()) - params.keys()
    assert not missing, f"Missing required NLC parameters in {xml_path}: {sorted(missing)}"

    params["num_samples"] = int(params["num_samples"])
    params["num_lines"] = int(params["num_lines"])
    return params


def read_nlc_raw(raw_path: str, params: dict,
                  frame_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Read frame data + timestamps from a Vevo F2 '.raw.nlc' file.

    Args:
        raw_path: path to the '.raw.nlc' file.
        params: dict from parse_nlc_xml() (needs 'num_samples', 'num_lines').
        frame_indices: 1-based frame numbers to load (matching the MATLAB reader's
            convention). None loads every frame.

    Returns:
        data: float32 array of shape (n_frames_loaded, num_samples, num_lines), linear
            (not log-compressed) amplitude values, axis 1 = depth/axial, axis 2 = width/lateral.
        timestamps_ms: float64 array of shape (n_frames_loaded,).
    """
    num_samples = params["num_samples"]
    num_lines = params["num_lines"]
    frame_data_bytes = 4 * num_samples * num_lines
    frame_block_bytes = FRAME_HEADER_BYTES + frame_data_bytes

    with open(raw_path, "rb") as f:
        version = struct.unpack("<I", f.read(4))[0]
        assert version >= 10, f"Unsupported file version {version} (expected Vevo F2, >= 10)"

        f.seek(4)
        num_frames = struct.unpack("<I", f.read(4))[0]

        f.seek(8)
        file_format = struct.unpack("<I", f.read(4))[0]
        assert not (file_format & RF_FORMAT_BIT), "Data is RF format, not RAW format"

        if frame_indices is None:
            frame_list = list(range(1, num_frames + 1))
        else:
            assert all(1 <= i <= num_frames for i in frame_indices), \
                f"frame_indices must be within 1..{num_frames}"
            frame_list = list(frame_indices)

        data = np.empty((len(frame_list), num_samples, num_lines), dtype=np.float32)
        timestamps_ms = np.empty(len(frame_list), dtype=np.float64)

        for out_ix, frame_num in enumerate(frame_list):
            frame_header_start = FILE_HEADER_BYTES + frame_block_bytes * (frame_num - 1)

            f.seek(frame_header_start + TIMESTAMP_OFFSET_IN_FRAME_HEADER)
            timestamps_ms[out_ix] = struct.unpack("<d", f.read(8))[0]

            f.seek(frame_header_start + FRAME_HEADER_BYTES)
            flat = np.frombuffer(f.read(frame_data_bytes), dtype="<f4")
            data[out_ix] = flat.reshape((num_lines, num_samples)).T

    return data, timestamps_ms
