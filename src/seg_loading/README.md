# Segmentation Loading

Segmentation loading plugins load binary masks to specify where in the loaded CEUS scan to analyze. Plugins support data loading from different file formats.

New plugins can be added to the [src/seg_loading/functions.py](functions.py) file as a new function, and will extend the capabilities of QuantUS without any additional programming required.

## Plugin Implementation

### Core data class

All segmentation parsers populate the `CeusSeg` class as defined in [src/data_objs/seg.py](../data_objs/seg.py).

```python
class CeusSeg:
    """
    Class for contrast-enhanced ultrasound image data.
    """

    def __init__(self):
        self.seg_name: str
        self.seg_mask: np.ndarray
        self.pixdim: List[float]  # voxel spacing in mm
```

The pixel dimensions, segmentation name, and binary mask are all that must be saved in a segmentation parser.

### Plugin Structure

Each segmentation loading plugin should be placed in the [src/seg_loading/functions.py](functions.py) file as a new function. Specifically, the new function must be in the following form:

```python
def SEG_LOADER_NAME(image_data: UltrasoundImage, seg_path: str, **kwargs) -> CeusSeg:
```

where `SEG_LOADER_NAME` is the name of your parser. The inputs contain the standard parser inputs for a segmentation parser, and the `kwargs` variable can be used to add any additional input variables that may be needed.

### Decorators

Metadata can be added to new segmentation parsing functions using decorators defined in [src/seg_loading/decorators.py](decorators.py).

Currently, the only implemented decorator for this parser is the `extensions` decorator, which specifies the required suffix of potential segmentation files.
