# Image Loading

Image loading plugins include all data loading parsers for QuantUS. These parsers can read CEUS scans from different manufacturers and file formats. Pixel data and linearized intensities for analysis are stored separately in these parsers for downstream flexibility.

New plugins can be dropped into this folder to extend the capabilities of QuantUS without any additional programming required.

## Plugin Implementation

### Core data class

All CEUS parsers load data into the `UltrasoundImage` class as defined in [src/data_objs/image.py](../data_objs/image.py). Thus, the class entrypoint of each parser is a child class of the `UltrasoundImage` class.

```python
class UltrasoundImage:
    """
    Class for general ultrasound image data (e.g., B-mode, CEUS, NIfTI).
    """

    def __init__(self, scan_path: str):
        self.scan_name = Path(scan_path).stem
        self.scan_path = scan_path
        self.pixel_data: np.ndarray # image data as a numpy array
        self.pixdim: List[float] # mm
        self.frame_rate: float # seconds/frame
        self.intensities_for_analysis: np.ndarray # linearized intensity values
        self.extras_dict: dict = {} # dictionary for any extra information inputted by plugins
```

For clarity, the `pixel_data` array contains uint8 grayscale values which are used for rendering the CEUS scan, with the last dimension indexing over time.

## Plugin Structure

Each image loading plugin should be placed in `src/image_loading/your_plugin_name/` with the following structure:

```
your_plugin_name/
├── main.py          # Required: EntryClass implementation
├── parser.py        # Recommended: Core parsing logic
├── objects.py       # Optional: Custom data structures
└── utils.py         # Optional: Helper functions
```

The child class of the `UltrasoundImage` base class should be named `EntryClass`, and this is the final class which will interact with the rest of the analysis workflow.

### Additional attributes

In addition to the default methods of the `UltrasoundImage` base class, the plugin `EntryClass` must also contain three additional class attributes to finish interfacing with the rest of the workflow.

| Attribute | Type | Description |
|-----------|------|-------------|
| `extensions` | List[str] | Supported file extensions (e.g., `[".bin", ".dat"]`) |
| `spatial_dims` | int | 2 for 2D data, 3 for 3D data |
| `required_kwargs` | List[str] | Required keyword arguments for initialization |

See the existing [nifti](nifti/) plugin for an example. Once implemented, this parser will be available from the GUI and CLI to run custom analysis with.
