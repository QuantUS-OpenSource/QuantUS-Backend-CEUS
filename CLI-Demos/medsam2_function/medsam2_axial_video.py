"""
Pull a fixed axial plane out of a 4D B-mode/CEUS pair, for inspecting contrast
and ROIs in a notebook.

    from medsam2_axial_video import suggest_z_from_voi, extract_axial_video

    z = suggest_z_from_voi(voi_path)
    video = extract_axial_video(bmode_path, ceus_path, z, n_frames=20)
"""

import nibabel as nib
import numpy as np


def suggest_z_from_voi(voi_path):
    """
    The reference VOI's centroid z index.

    Picking the centre slice rather than an arbitrary index maximises how long
    the lesion stays inside the fixed plane as it drifts with respiration,
    which is the main thing limiting how many frames are usable.
    """
    voi = np.asarray(nib.load(voi_path).dataobj) > 0
    if not voi.any():
        raise ValueError(f"Reference VOI is empty: {voi_path!r}")
    return int(round(np.argwhere(voi)[:, 2].mean()))


def extract_axial_video(bmode_path, ceus_path, z, frame_indices=None,
                        n_frames=20, t_start=0, t_stop=None,
                        site=None, patient_number=None, visit=None, bolus=None):
    """
    One fixed axial plane through a 4D B-mode/CEUS pair, across time.

    Returns a dict with `bmode` and `ceus` as (T, H, W) uint8 arrays, plus the
    source frame indices, the z index, mm spacing and volume rate. Slices are
    oriented (Y, X) to match `medsam2_viewer_base.get_plane_slice(vol,
    "Axial", z)`, so a mask drawn here indexes the same way as everywhere else
    in this package.

    `frame_indices` selects the frames explicitly (e.g. the phase-anchored set
    from `extract_temporal_dataset`); otherwise `n_frames` are spread evenly
    over [t_start, t_stop). Only the requested planes are read -- the source
    series are several GB and are never loaded whole.

    `site`/`patient_number`/`visit`/`bolus` are carried through untouched (pass
    the master CSV row's fields). They are what `AxialRoiAnnotator` uses to
    look the case's canonical subject_id/sequence_id out of the fine-tuning
    manifest, so supplying them is what makes this set line up with
    MedSAM2_finetune_data.
    """
    bmode_img = nib.load(bmode_path)
    ceus_img = nib.load(ceus_path)

    if bmode_img.shape[:3] != ceus_img.shape[:3]:
        raise ValueError(f"B-mode {bmode_img.shape} and CEUS {ceus_img.shape} differ in space")
    n_total = min(bmode_img.shape[-1], ceus_img.shape[-1])
    n_z = bmode_img.shape[2]
    if not (0 <= z < n_z):
        raise ValueError(f"z={z} out of range [0, {n_z})")

    if frame_indices is None:
        t_stop = n_total if t_stop is None else min(int(t_stop), n_total)
        frame_indices = np.unique(
            np.linspace(int(t_start), t_stop - 1, int(n_frames)).round().astype(int)
        ).tolist()
    else:
        frame_indices = [int(t) for t in frame_indices]
        bad = [t for t in frame_indices if not (0 <= t < n_total)]
        if bad:
            raise ValueError(f"frame indices out of range [0, {n_total}): {bad}")

    def _stack(img):
        # (X, Y) per frame -> transposed to (Y, X), matching get_plane_slice
        return np.stack(
            [np.asarray(img.dataobj[:, :, z, t], dtype=np.uint8).T for t in frame_indices]
        )

    zooms = bmode_img.header.get_zooms()
    rate_hz = float(zooms[3]) if len(zooms) > 3 and zooms[3] > 0 else None

    return {
        "bmode": _stack(bmode_img),
        "ceus": _stack(ceus_img),
        "frame_indices": frame_indices,
        "z": int(z),
        "n_source_frames": int(n_total),
        # in-plane spacing is (dy, dx) because the slices are transposed
        "spacing_mm_yx": (float(zooms[1]), float(zooms[0])),
        "spacing_mm_xyz": [float(v) for v in zooms[:3]],
        "frame_rate_hz": rate_hz,
        "time_s": [(t / rate_hz) if rate_hz else None for t in frame_indices],
        "bmode_path": bmode_path,
        "ceus_path": ceus_path,
        "site": site,
        "patient_number": patient_number,
        "visit": visit,
        "bolus": bolus,
    }
