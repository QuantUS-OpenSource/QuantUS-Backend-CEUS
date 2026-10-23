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
| `case_files.py` | locate a case's B-mode / CEUS / VOI files from the CSV |

## Setup

`sam2` is a normal installed package, not a folder in this repo:

```bash
pip install "git+https://github.com/Yuanshan-Wu/MedSAM2.git@3DMPUS_finetune"
```

Checkpoints are not stored in git. The notebooks read the checkpoint path from
the `MEDSAM2_CHECKPOINT` environment variable and otherwise default to the
pretrained `MedSAM2_MRI_LiverLesion.pt`.

## Training and evaluation

Fine-tuning, dataset preparation, k-fold splits and the quantitative evaluation
live in the **3DMPUS-segmentation** repository. Use a fine-tuned checkpoint here by
pointing `MEDSAM2_CHECKPOINT` at its `.pt` file.
