# MedSAM2 inference helpers

Segmentation of the tracked tumor volume with MedSAM2, used by the inference
notebooks in `CLI-Demos/` (`MedSam2_inference.ipynb`, `Batch_MedSAM2_inference.ipynb`,
`MedSam2_fineTuning_extract.ipynb` for the 2D evaluation).

| File | Purpose |
|---|---|
| `medsam2_3d_mask.py` | `Medsam2AdaptiveBboxMasker`: per-frame 3D mask from an adaptive box prompt |
| `medsam2_axial_propagation.py` | axial-propagation variant |
| `medsam2_napari_viewer.py` | napari 4D viewer |
| `medsam2_viewer_base.py` | shared mask/denoising/Dice helpers |
| `medsam2_2d_temporal_roi.py` | 2D fixed-slice temporal ROI evaluation |
| `fold_checkpoints.py` | pick the cross-validation checkpoint that never saw a given patient |
| `seg_metrics.py` | Dice-family / surface-distance metrics (MONAI) for the 2D evaluation |
| `case_files.py` | locate a case's B-mode / CEUS / VOI files from the CSV |

## Setup

`sam2` is a normal installed package, not a folder in this repo:

```bash
pip install "git+https://github.com/Yuanshan-Wu/MedSAM2.git@3DMPUS_finetune"
```

Checkpoints are not stored in git. The notebooks segment each patient with the
cross-validation checkpoint of the fold that held that patient out
(`AI_inference/MedSAM2/checkpoints/fold_{1..5}.pt`, see `AI_inference/MedSAM2/README.md`).
Set the `MEDSAM2_CHECKPOINT` environment variable to force one checkpoint instead.

## Training and evaluation

Fine-tuning, dataset preparation, k-fold splits and the quantitative evaluation
live in the **3DMPUS-segmentation** repository. Use a fine-tuned checkpoint here by
pointing `MEDSAM2_CHECKPOINT` at its `.pt` file.
