# Segmentation Loading

Segmentation preprocessing plugins specify transformations to apply to the loaded binary segmentation mask before starting with analysis. These transformations are typically there to align the segmentation mask with the preprocessed image.

New plugins can be added to the [src/seg_preprocessing/functions.py](functions.py) file as a new function, and will extend the capabilities of QuantUS without any additional programming required.

## Plugin Implementation

### Plugin Structure

Each segmentation loading plugin should be placed in the [src/seg_preprocessing/functions.py](functions.py) file as a new function. Specifically, the new function must be in the following form:

```python
def SEG_PREPROC_NAME(image_data: UltrasoundImage, seg_data: CeusSeg, **kwargs) -> CeusSeg:
```

where `SEG_PREPROC_NAME` is the name of your preprocessing step. Outside of the standard variable inputs, the `kwargs` variable can be used to add any additional input variables that may be needed.

### Decorators

Metadata can be added to new segmentation parsing functions using decorators defined in [src/seg_preprocessing/decorators.py](decorators.py).

Currently, the only implemented decorator for this parser is the `required_kwargs` decorator, which lists the additional variables needed for a preprocessing step outside of the standard inputs.
