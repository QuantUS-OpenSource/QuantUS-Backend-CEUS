# Curve Loading

Curve loading plugins specify how to load pre-computed CEUS curves to use for downstream analysis.

New plugins can be added to the [src/curve_loading/functions.py](functions.py) file as a new function, and will extend the capabilities of QuantUS without any additional programming required.

## Plugin Implementation

### Plugin Structure

Each segmentation loading plugin should be placed in the [src/curve_loading/functions.py](functions.py) file as a new function. Specifically, the new function must be in the following form:

```python
def CURVE_LOADER(curves_path: str, **kwargs) -> CurvesAnalysis:
```

where `CURVE_LOADER` is the name of your curve loading plugin. The `curves_path` input is the standard input here, but the `kwargs` variable can be used to add any additional input variables that may be needed.
