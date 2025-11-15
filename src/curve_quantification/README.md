# Curve Quantification

Curve quantification plugins specify how to compute individual parameters from CEUS curves computed earlier in the analysis pipeline.

New plugins can be added to the [src/curve_quantification/functions.py](functions.py) file as a new function, and will extend the capabilities of QuantUS without any additional programming required.

## Plugin Implementation

### Plugin Structure

Each segmentation loading plugin should be placed in the [src/curve_quantification/functions.py](functions.py) file as a new function. Specifically, the new function must be in the following form:

```python
def QUANT_NAME(analysis_objs: CurvesAnalysis, curves: Dict[str, List[float]], data_dict: dict,  **kwargs) -> None:
```

where `QUANT_NAME` is the name of your curve quantification plugin. The inputs contain the standard curve quantification inputs, and the `kwargs` variable can be used to add any additional input variables that may be needed.

* The `curves` dictionary is organized with the string name of the curve as the key, and the array containing the appropriate curve as the value.
* The `data_dict` dictionary is used to store all quantified values from the curves. Each plugin should populate outputs into this dict with the key being the parameter name and the value being a numerical output.

### Decorators

Metadata can be added to new segmentation parsing functions using decorators defined in [src/curve_quantification/decorators.py](decorators.py).

* The `required_kwargs` decorator lists the additional variables needed for a curve definition plugin. Detailed explanations of each kwarg should be listed in the docstring of the plugin function.
* The `dependencies` decorator lists the other plugins which must be run before the current plugin. The results of previously run curve quantification functions are accessible via the `data_dict` dictionary.
