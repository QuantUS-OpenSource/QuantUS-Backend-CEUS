# Visualizations

If parametric map analysis has been used, this step can export parametric map outputs and visualizations. This module is extendable to support visualizations from non parametric map analysis as well.

By default, parametric maps are exported in array form as `.npy` files, with each element containing direct parameter values organized spatially. See `CLI-Demos/paramaps_viewing_demo.ipynb` for more details.

New simple visualizations can be added to the [src/visualizations/paramap/functions.py](curve_types/functions.py) file as a new function, and will extend the capabilities of QuantUS without any additional programming required.

## Plugin Implementation

### Plugin Structure

Each curve definition plugin should be placed in the [src/visualizations/paramap/functions.py](functions.py) file as a new function. Specifically, the new function must be in the following form:

```python
def VIS_NAME(quants_obj: CurveQuantifications, paramap_folder_path: str, **kwargs):
```

where `VIS_NAME` is the name of your visualization plugin. The inputs contain the standard parser inputs for a segmentation parser, and the `kwargs` variable can be used to add any additional input variables that may be needed.

* The `paramap_folder_path` input contains the name of the folder in which all visualizations should be exported to.
