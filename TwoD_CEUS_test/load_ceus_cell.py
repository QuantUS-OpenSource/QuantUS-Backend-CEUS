# ============================================================================
# LOAD CEUS DATA
# ============================================================================
import nibabel as nib
import numpy as np

# Load the CEUS data
ceus_path = '/Users/samantha/Desktop/ultrasound lab stuff/raw_ctdna/p14/wk12/combined_nifti/p14_wk12_CHI_RAW.nii'
print(f"Loading CEUS data from: {ceus_path}")

ceus_data = nib.load(ceus_path).get_fdata()

print(f"CEUS data shape: {ceus_data.shape}")
print(f"CEUS data dtype: {ceus_data.dtype}")
print(f"CEUS data range: [{ceus_data.min():.6f}, {ceus_data.max():.6f}]")
