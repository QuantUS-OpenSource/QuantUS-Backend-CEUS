# Time series analysis

Inspired by time-intensity curve (TIC) analysis, analysis in this workflow is based around quantifying each frame in a CEUS scan, and stringing these numbers together to create quantitative curve representations of the CEUS data. Multiple curves can be generated within a single analysis run, and these curve definitions are customizable.

QuantUS supports whole segmentation and parametric map CEUS analysis.

New curve definitions can be added to the [src/time_series_analysis/curve_types/functions.py](curve_types/functions.py) file as a new function, and will extend the capabilities of QuantUS without any additional programming required.

## Structure

* `curve_types` folder - contains all definitions of curves that can potentially be analyzed. Plugins for different curve definitions are located here.
* `curves` folder - contains core architecture which interacts with selected curve definitions, computes curves, and saves them. These curves are computed on the whole segmentation. Curves themselves can be optionally be exported.
* `curves_paramaps` folder - contains core architecture which interacts with selected curve definitions, computes curves, and saves them. These curves are computed on different sections of the segmentation using the sliding window technique, so they can be used to create parametric maps downstream in the analysis pipeline. All curves themselves can be optionally exported.

## Plugin Implementation

### Plugin Structure

Each curve definition plugin should be placed in the [src/time_series_analysis/curve_types/functions.py](functions.py) file as a new function. Specifically, the new function must be in the following form:

```python
def CURVE_NAME(image_data: UltrasoundImage, frame: np.ndarray, mask: np.ndarray, **kwargs) -> Tuple[List[str], List[np.ndarray]]:
```

where `CURVE_NAME` is the name of your parser. The inputs contain the standard inputs for a curve type function, and the `kwargs` variable can be used to add any additional input variables that may be needed.

* The `frame` input contains a single frame of linearized intensities of the inputted CEUS scan
* The `mask` input contains the binary segmentation mask
* The first element of the returned tuple is the addressable name of each curve that has been generated. The corresponding arrays in the second element of the returned tuple is the values of the curve of each name. Both elements of the returned tuples should be lists of equal length.

### Decorators

Metadata can be added to new segmentation parsing functions using decorators defined in [src/time_series_analysis/curve_types/decorators.py](decorators.py).

* The `required_kwargs` decorator lists the additional variables needed for a curve definition plugin. Detailed explanations of each kwarg should be listed in the docstring of the plugin function.
