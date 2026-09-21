"""
Fixed-z axial ROI annotation over time, for the 2D temporal evaluation set.

One axial plane (a single z index) is pulled out of the 4D B-mode and CEUS
series across a chosen set of frames, giving a (T, H, W) "video" of that one
plane. `AxialRoiAnnotator` then lets you draw a closed ROI on each frame in
the notebook and writes the images plus the masks to disk.

What this measures, and what it does not
----------------------------------------
The lesion moves in all three axes with respiration, so a fixed z index does
not follow the same anatomical cut over time -- at some frames the plane runs
through the middle of the lesion, at others it clips an edge or misses it.
Because the ROI is redrawn on the image actually shown at each frame, the
mask and the image always agree and Dice stays a fair per-frame score; out-of-
plane motion shows up as *variation in the test set*, not as a bias in the
metric. Two consequences to carry into the analysis:

  - Frames where the lesion has left the plane have an empty ROI. Those are
    kept (with an all-zero mask, `annotated` still True) because they are the
    false-positive test; Dice is undefined there and must be reported
    separately from the mean, not silently dropped.
  - Dice is biased against small objects, and the lesion's cross-section
    shrinks as the plane drifts off-centre. `gt_area_px` is saved per frame so
    a dip in Dice can be checked against cross-section size before it is read
    as temporal degradation.

Also note this scores 2D single-plane segmentation, which is a different
inference mode from the axial-propagation pipeline the fine-tuned checkpoint
is used through (medsam2_axial_propagation.py). A single axial cut carries no
information about propagation extent along z, since z is the through-plane
axis here.

Usage (in the notebook)
-----------------------
    %matplotlib widget        # required -- ipympl backend, for mouse events

    from medsam2_2d_temporal_roi import (
        extract_axial_video, suggest_z_from_voi, AxialRoiAnnotator,
    )

    z = suggest_z_from_voi(voi_path)
    video = extract_axial_video(bmode_path, ceus_path, z, n_frames=20)

    ann = AxialRoiAnnotator(video, subject_id="subject_000001",
                            output_root="/.../MedSAM2_2D_temporal_eval")
    ann.show()

Drawing: left-click adds a point, right-click removes the last one. Three
points are enough to close a contour; the ROI fills live as you add more.
"Copy prev" seeds the current frame with the previous frame's points so a
slowly drifting lesion only needs nudging rather than redrawing.
"""

import csv
import glob
import json
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from ipywidgets import (
    Button, Checkbox, HBox, HTML, IntSlider, Layout, ToggleButtons, VBox,
)
from IPython.display import display
from matplotlib.path import Path as MplPath
from scipy.interpolate import splev, splprep

from extract_finetune_dataset import derive_native_subject_id, find_voi_file
from medsam2_viewer_base import (
    dice_score, enhance_bmode_noise, get_mask_boundary, slice_to_rgb,
)


# ─────────────────────────────────────────────────────────────────────────────
# Identity — matching a case back to MedSAM2_finetune_data
# ─────────────────────────────────────────────────────────────────────────────

def lookup_case_ids(finetune_manifest, site, patient_number, visit, bolus,
                    bmode_path=None):
    """
    The canonical `subject_id` / `sequence_id` for one case, read out of the
    fine-tuning `manifest.csv`.

    Those ids must never be typed by hand. Neither is guessable from the case
    itself, and both fail quietly when guessed wrong:

      - `subject_XXXXXX` is a positional index over `sorted({(site, patient)})`
        (`extract_finetune_dataset.build_manifest`), so UCSD-P05 is
        subject_000009 for no reason visible from the case.
      - `sequence_XXX` indexes the patient's sorted (visit, bolus) pairs, and
        bolus labels sort lexicographically -- TJU-P06's sequence_001 is
        V01/**CE2**, not V01/CEUS1, because "CE2" < "CEUS1". Reading the ids
        off the bolus you happen to be annotating gets this backwards.

    Matched on site + patient + visit + bolus, which is exactly the
    `matching_basis` the fine-tuning image.json records for itself; the
    manifest's stored `bmode_path` is deliberately *not* the key, because
    source files get renamed and those paths go stale. Passing `bmode_path`
    cross-checks against it and warns on a mismatch rather than failing.
    """
    with open(finetune_manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    key = (str(site).strip(), str(patient_number).strip(),
           str(visit).strip(), str(bolus).strip())
    matches = [
        r for r in rows
        if (r["site"].strip(), r["patient_number"].strip(),
            r["visit"].strip(), r["bolus"].strip()) == key
    ]
    if len(matches) != 1:
        raise LookupError(
            f"Expected exactly 1 row in {finetune_manifest} for "
            f"site/patient/visit/bolus={key}, found {len(matches)}"
        )
    row = matches[0]

    if bmode_path and row["bmode_path"] and \
            os.path.basename(bmode_path) != os.path.basename(row["bmode_path"]):
        print(f"[lookup] note: source file renamed since the manifest was written\n"
              f"         manifest: {os.path.basename(row['bmode_path'])}\n"
              f"         actual:   {os.path.basename(bmode_path)}")

    # Derived from the file actually being annotated, not the manifest's
    # possibly-stale path, so a renamed source yields the current native id.
    native = derive_native_subject_id(bmode_path or row["bmode_path"])

    return {
        "subject_id": row["subject_id"],
        "sequence_id": row["sequence_id"],
        "native_subject_id": native,
        "canonical_scan_id": f"3DMPUS|{native}||Ultrasound",
        "dataset": "3DMPUS",
        "site": row["site"],
        "patient_number": row["patient_number"],
        "visit": row["visit"],
        "bolus": row["bolus"],
        "finetune_frame_idx": (int(row["frame_idx"])
                               if row.get("frame_idx", "").strip() else None),
        "matching_basis": "site, patient number, visit, and bolus",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Contour -> mask
# ─────────────────────────────────────────────────────────────────────────────

def closed_spline(points, n_out=400):
    """
    Smooth closed curve through the clicked points.

    Matches how QuantUS itself turns clicks into a contour (a periodic
    `splprep`/`splev` pass, `src/ceus/seg_loading/views/spline.py`) so ROIs
    drawn here have the same character as ones drawn in ceus_gui.py, rather
    than being raw polygons. Falls back to the polygon if the spline fit fails
    -- collinear or near-duplicate clicks can make `splprep` singular, and a
    slightly angular ROI beats losing the annotation.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts

    # splprep needs strictly distinct knots; consecutive duplicate clicks are
    # common when double-clicking.
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
    pts = pts[keep]
    if len(pts) < 3:
        return pts

    ring = np.vstack([pts, pts[:1]])
    try:
        tck, _ = splprep([ring[:, 0], ring[:, 1]], s=0.0, per=True,
                         k=min(3, len(pts) - 1))
        x, y = splev(np.linspace(0, 1, n_out), tck)
        return np.column_stack([x, y])
    except Exception:
        return ring


def points_to_mask(points, shape):
    """Rasterize clicked points into a filled binary mask of `shape` (H, W)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if len(points) < 3:
        return mask

    curve = closed_spline(points)
    yy, xx = np.mgrid[:h, :w]
    inside = MplPath(curve).contains_points(
        np.column_stack([xx.ravel(), yy.ravel()])
    )
    return inside.reshape(h, w)


# ─────────────────────────────────────────────────────────────────────────────
# Annotator
# ─────────────────────────────────────────────────────────────────────────────

class AxialRoiAnnotator:
    """
    Draw one ROI per frame on a fixed axial plane, then save.

    Saving is incremental and idempotent: only frames that have been drawn on
    are marked annotated, `save()` can be pressed at any point, and
    reconstructing the annotator with `resume=True` reloads whatever was saved
    so a session can be picked up later. Contour *points* are stored alongside
    the rasterized masks, so a saved ROI can be reopened and adjusted rather
    than redrawn from scratch.
    """

    def __init__(self, video, output_root, subject_id=None, sequence_id=None,
                 finetune_manifest=None, case_ids=None, resume=True,
                 figsize=(7.5, 6.5), extra_meta=None):
        self.video = video
        self.output_root = output_root
        self.extra_meta = dict(extra_meta or {})

        # Identity comes from the fine-tuning manifest whenever one is given,
        # so this set lines up with MedSAM2_finetune_data by construction
        # rather than by a hand-typed string -- see `lookup_case_ids` for why
        # guessing these is unsafe. Explicit subject_id/sequence_id still win,
        # for cases that are not in the manifest at all.
        identifiers = [video.get(k) for k in ("site", "patient_number", "visit", "bolus")]
        if case_ids is None and finetune_manifest is not None and all(identifiers):
            case_ids = lookup_case_ids(
                finetune_manifest,
                site=video["site"], patient_number=video["patient_number"],
                visit=video["visit"], bolus=video["bolus"],
                bmode_path=video["bmode_path"],
            )
        self.case_ids = dict(case_ids or {})
        if self.case_ids:
            subject_id = subject_id or self.case_ids["subject_id"]
            sequence_id = sequence_id or self.case_ids["sequence_id"]
        if subject_id is None or sequence_id is None:
            raise ValueError(
                "Need subject_id/sequence_id. Pass finetune_manifest= together with "
                "site/patient_number/visit/bolus on extract_axial_video() to resolve "
                "them from MedSAM2_finetune_data/manifest.csv, or set them explicitly."
            )
        self.subject_id = subject_id
        self.sequence_id = sequence_id

        self.bmode = video["bmode"]
        self.ceus = video["ceus"]
        self.z = video["z"]
        self.frame_indices = list(video["frame_indices"])
        self.n_frames, self.height, self.width = self.bmode.shape

        self.case_id = f"{self.subject_id}_{self.sequence_id}_z{self.z:04d}"
        # Mirrors MedSAM2_finetune_data's processed/{subject}/{sequence}/ layout,
        # with the slice index appended the same way the 3D temporal set appends
        # its frame index, so the trees line up under a common glob.
        self.case_dir = os.path.join(
            output_root, "processed", self.subject_id, f"{self.sequence_id}_z{self.z:04d}"
        )

        # frame index -> list of (x, y) clicks
        self.points = {t: [] for t in range(self.n_frames)}
        self.annotated = np.zeros(self.n_frames, dtype=bool)
        self._display_cache = {}

        if resume:
            self._load_existing()

        self._build_ui(figsize)

    # ── persistence ────────────────────────────────────────────────────────

    def _load_existing(self):
        path = os.path.join(self.case_dir, f"{self.case_id}.json")
        if not os.path.isfile(path):
            return
        with open(path) as f:
            saved = json.load(f)
        if saved.get("frame_indices") != self.frame_indices:
            print(f"[resume] skipped — saved frame set differs from this video's ({path})")
            return
        for entry in saved.get("frames", []):
            t = entry["t"]
            if 0 <= t < self.n_frames and entry.get("annotated"):
                self.points[t] = [tuple(p) for p in entry.get("points", [])]
                self.annotated[t] = True
        print(f"[resume] loaded {int(self.annotated.sum())} annotated frame(s) from {path}")

    def save(self, _button=None):
        """
        Write images, masks and contour points for every annotated frame.

        `imgs`/`gts` follow the key names `build_npz_dataset.py` writes and
        MedSAM2's NPZRawDataset reads, with time in the axis that normally
        holds z -- so the saved stack loads with the existing tooling.
        """
        os.makedirs(self.case_dir, exist_ok=True)
        idx = np.flatnonzero(self.annotated)
        if len(idx) == 0:
            self._set_status("Nothing to save — no frames annotated yet.", warn=True)
            return

        masks = np.zeros((self.n_frames, self.height, self.width), dtype=np.uint8)
        for t in idx:
            masks[t] = points_to_mask(self.points[t], (self.height, self.width))

        stem = os.path.join(self.case_dir, self.case_id)
        np.savez_compressed(
            f"{stem}.npz",
            imgs=self.bmode[idx],
            imgs_ceus=self.ceus[idx],
            gts=masks[idx],
            frame_indices=np.array([self.frame_indices[t] for t in idx], dtype=np.int32),
        )

        # (H, W, T) NIfTI triplet for visual QC in ITK-SNAP/Slicer. Spacing is
        # the in-plane mm pair; the third axis is time, left at 1.0 since it is
        # not a spatial dimension.
        dy, dx = self.video["spacing_mm_yx"]
        affine = np.diag([dy, dx, 1.0, 1.0])
        for name, arr in (("bmode", self.bmode[idx]), ("ceus", self.ceus[idx]),
                          ("mask", masks[idx])):
            img = nib.Nifti1Image(np.transpose(arr, (1, 2, 0)), affine)
            img.header.set_xyzt_units("mm")
            nib.save(img, f"{stem}_{name}.nii.gz")

        # Identity fields use the same names and meanings as the fine-tuning
        # image.json: `subject_id` is the human-readable native id, the
        # standardized_* pair carries the folder-name ids. A case annotated
        # without a manifest lookup falls back to the standardized id for both.
        meta = {
            "case_id": self.case_id,
            "dataset": self.case_ids.get("dataset", "3DMPUS"),
            "subject_id": self.case_ids.get("native_subject_id", self.subject_id),
            "standardized_subject_id": self.subject_id,
            "standardized_sequence_id": f"{self.sequence_id}_z{self.z:04d}",
            "sequence_id": self.sequence_id,
            "canonical_scan_id": self.case_ids.get("canonical_scan_id"),
            "matching_basis": self.case_ids.get(
                "matching_basis", "site, patient number, visit, and bolus"),
            "site": self.case_ids.get("site"),
            "patient_number": self.case_ids.get("patient_number"),
            "visit": self.case_ids.get("visit"),
            "bolus": self.case_ids.get("bolus"),
            # the single frame this subject was fine-tuned on, so the temporal
            # curve can be read relative to the model's training timepoint
            "finetune_frame_idx": self.case_ids.get("finetune_frame_idx"),
            "schema": "3DMPUS.axial2D_temporal_eval.v1",
            "recommended_split": "eval_temporal_2d",
            "z": self.z,
            "plane": "Axial",
            "frame_indices": self.frame_indices,
            "n_source_frames": self.video["n_source_frames"],
            "frame_rate_hz": self.video["frame_rate_hz"],
            "spacing_mm_yx": list(self.video["spacing_mm_yx"]),
            "spacing_mm_xyz": self.video["spacing_mm_xyz"],
            "shape_hw": [self.height, self.width],
            "bmode_path": self.video["bmode_path"],
            "ceus_path": self.video["ceus_path"],
            "saved_frame_positions": [int(t) for t in idx],
            "n_annotated": int(len(idx)),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "frames": [
                {
                    "t": int(t),
                    "source_frame_index": self.frame_indices[t],
                    "time_s": self.video["time_s"][t],
                    "annotated": bool(self.annotated[t]),
                    "points": [[float(x), float(y)] for x, y in self.points[t]],
                    # empty ROI = lesion out of plane; kept deliberately as the
                    # false-positive test rather than dropped
                    "gt_area_px": int(masks[t].sum()),
                    "gt_area_mm2": float(masks[t].sum() * dy * dx),
                }
                for t in range(self.n_frames)
            ],
            **self.extra_meta,
        }
        with open(f"{stem}.json", "w") as f:
            json.dump(meta, f, indent=2)

        empty = sum(1 for t in idx if masks[t].sum() == 0)
        self._set_status(
            f"Saved {len(idx)} frame(s)"
            + (f" ({empty} with an empty ROI)" if empty else "")
            + f" → {self.case_dir}"
        )

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self, figsize):
        self.status = HTML(value="")

        self.frame_slider = IntSlider(
            value=0, min=0, max=self.n_frames - 1, step=1, description="Frame:",
            continuous_update=False, layout=Layout(width="520px"),
        )
        self.frame_slider.observe(lambda _c: self._render(), names="value")

        self.modality_toggle = ToggleButtons(
            options=["B-mode", "CEUS"], value="B-mode", description="Show:",
        )
        self.modality_toggle.observe(lambda _c: self._render(), names="value")

        self.enhance_checkbox = Checkbox(value=False, description="Enhance display")
        self.enhance_checkbox.observe(lambda _c: self._render(), names="value")

        self.ghost_checkbox = Checkbox(value=True, description="Show previous frame's ROI")
        self.ghost_checkbox.observe(lambda _c: self._render(), names="value")

        def button(desc, handler, tooltip, style=""):
            b = Button(description=desc, tooltip=tooltip, button_style=style,
                       layout=Layout(width="120px"))
            b.on_click(handler)
            return b

        buttons = HBox([
            button("◀ Prev", lambda _b: self._step(-1), "Previous frame"),
            button("Next ▶", lambda _b: self._step(+1), "Next frame"),
            button("Undo point", self._undo, "Remove the last clicked point"),
            button("Clear frame", self._clear, "Discard this frame's ROI"),
            button("Copy prev", self._copy_prev, "Seed with the previous frame's points"),
            button("Mark empty", self._mark_empty,
                   "Lesion not in this plane — record an empty ROI"),
            button("Save", self.save, "Write annotated frames to disk", "success"),
        ])

        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        plt.ion()

        self.ui = VBox([
            HBox([self.frame_slider, self.modality_toggle]),
            HBox([self.enhance_checkbox, self.ghost_checkbox]),
            buttons,
            self.status,
            self.fig.canvas,
        ])

    def _step(self, delta):
        self.frame_slider.value = int(
            np.clip(self.frame_slider.value + delta, 0, self.n_frames - 1)
        )

    def _on_click(self, event):
        # The pan/zoom tools also emit button_press_event; ignore clicks while
        # one is active so navigating the image doesn't drop stray points.
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if getattr(toolbar, "mode", "") not in ("", None):
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return

        t = self.frame_slider.value
        if event.button == 3:                     # right-click removes
            if self.points[t]:
                self.points[t].pop()
        elif event.button == 1:
            self.points[t].append((float(event.xdata), float(event.ydata)))
        else:
            return

        self.annotated[t] = bool(self.points[t])
        self._render()

    def _undo(self, _b):
        t = self.frame_slider.value
        if self.points[t]:
            self.points[t].pop()
            self.annotated[t] = bool(self.points[t])
            self._render()

    def _clear(self, _b):
        t = self.frame_slider.value
        self.points[t] = []
        self.annotated[t] = False
        self._render()

    def _copy_prev(self, _b):
        t = self.frame_slider.value
        prev = next((p for p in range(t - 1, -1, -1) if self.annotated[p]), None)
        if prev is None:
            self._set_status("No earlier annotated frame to copy from.", warn=True)
            return
        self.points[t] = list(self.points[prev])
        self.annotated[t] = bool(self.points[t])
        self._render()

    def _mark_empty(self, _b):
        """Record "lesion not in this plane" — annotated, but with no ROI."""
        t = self.frame_slider.value
        self.points[t] = []
        self.annotated[t] = True
        self._render()

    def _display_slice(self, t):
        key = (t, self.modality_toggle.value, self.enhance_checkbox.value)
        if key not in self._display_cache:
            arr = (self.bmode if self.modality_toggle.value == "B-mode" else self.ceus)[t]
            self._display_cache[key] = (
                enhance_bmode_noise(arr) if self.enhance_checkbox.value
                else slice_to_rgb(arr)[..., 0]
            )
        return self._display_cache[key]

    def _set_status(self, message, warn=False):
        colour = "#b36b00" if warn else "#333"
        n_done = int(self.annotated.sum())
        self.status.value = (
            f"<span style='color:{colour}'>{message}</span> &nbsp;|&nbsp; "
            f"<b>{n_done}/{self.n_frames}</b> frames annotated"
        )

    def _render(self):
        t = self.frame_slider.value
        pts = self.points[t]
        mask = points_to_mask(pts, (self.height, self.width))

        self.ax.clear()
        self.ax.imshow(self._display_slice(t), cmap="gray")

        if self.ghost_checkbox.value:
            prev = next((p for p in range(t - 1, -1, -1) if self.annotated[p]), None)
            if prev is not None and len(self.points[prev]) >= 3:
                ghost = closed_spline(self.points[prev])
                self.ax.plot(ghost[:, 0], ghost[:, 1], color="deepskyblue",
                             lw=1.0, ls=":", alpha=0.8)

        if mask.any():
            self.ax.contour(get_mask_boundary(mask), colors="red", linewidths=2)
        if len(pts) >= 3:
            curve = closed_spline(pts)
            self.ax.plot(curve[:, 0], curve[:, 1], color="red", lw=1.0, alpha=0.6)
        if pts:
            xs, ys = zip(*pts)
            self.ax.plot(xs, ys, "o", color="yellow", ms=4, mec="black", mew=0.5)

        source_t = self.frame_indices[t]
        time_s = self.video["time_s"][t]
        stamp = f"{time_s:.1f}s" if time_s is not None else f"frame {source_t}"
        flag = "✓" if self.annotated[t] else "·"
        self.ax.set_title(
            f"{flag} {self.subject_id} — z={self.z}, frame {source_t} ({stamp})  "
            f"[{t + 1}/{self.n_frames}]", fontsize=11,
        )
        self.ax.set_xlabel("Lateral (X)")
        self.ax.set_ylabel("Depth (Y)")
        self.fig.canvas.draw_idle()

        dy, dx = self.video["spacing_mm_yx"]
        area = mask.sum()
        detail = (f"{len(pts)} point(s), area {area} px ({area * dy * dx:.1f} mm²)"
                  if pts else
                  ("marked empty (lesion out of plane)" if self.annotated[t]
                   else "not annotated"))
        self._set_status(f"Frame {source_t}: {detail}")

    def show(self):
        display(self.ui)
        self._render()


# ─────────────────────────────────────────────────────────────────────────────
# Scoring against Medsam2AdaptiveBboxMasker
# ─────────────────────────────────────────────────────────────────────────────

def load_2d_case(case_dir):
    """The saved metadata for one annotated case (the `*.json` in `case_dir`)."""
    candidates = [f for f in os.listdir(case_dir)
                  if f.endswith(".json") and not f.endswith("_meta.json")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly 1 metadata JSON in {case_dir!r}, found {candidates!r}")
    with open(os.path.join(case_dir, candidates[0])) as f:
        return json.load(f)


NSD_TOLERANCE_MM = 1.0
METRICS = ("dice", f"nsd@{NSD_TOLERANCE_MM}mm", "assd_mm", "hd95_mm")

# Higher is better for overlap metrics, lower for the millimetre distances --
# the plot needs this to point its "better" arrow the right way.
METRIC_HIGHER_IS_BETTER = {
    "dice": True, f"nsd@{NSD_TOLERANCE_MM}mm": True,
    "iou": True, "assd_mm": False, "hd95_mm": False,
}
METRIC_LABELS = {
    "dice": "Dice", f"nsd@{NSD_TOLERANCE_MM}mm": f"NSD @ {NSD_TOLERANCE_MM} mm",
    "iou": "IoU", "assd_mm": "ASSD (mm)", "hd95_mm": "HD95 (mm)",
}


def evaluate_2d_case(masker, case_dir, verbose=True, return_masks=True):
    """
    Score `Medsam2AdaptiveBboxMasker` predictions against the drawn 2D GT,
    one row per annotated frame.

    The prediction for a frame is the fixed-z axial slice of that frame's full
    adaptive-bbox reconstruction: `compute_frame(t)["mask_3d"][:, :, z].T`,
    which is (Y, X) and so lines up with the GT drawn by `AxialRoiAnnotator`
    and with `get_plane_slice(volume, "Axial", z)`.

    Why this comparison is legitimate on *empty* frames, which is the whole
    reason for annotating them: `compute_frame` prompts from
    `seg_data.motion_compensation.tracked_bboxes[frame_idx]`, so the box comes
    from motion compensation and never from the GT. Contrast
    `medsam2_gt_eval.bbox_from_mask`, which derives its box from the GT mask
    and therefore cannot score a frame whose GT is empty at all.

    Each row carries the `gt_mask` / `pred_mask` arrays that `plot_2d_overlay`
    draws, unless `return_masks=False`. They cost about 2.5 MB for a 20-frame
    case, which is not worth making the caller opt in for and then hit a
    failure two cells later.

    Rows are not pooled into one number. Dice is undefined when the GT is
    empty (`dice_score` returns NaN when both masks are empty and 0 when only
    the prediction is non-empty), so averaging over all frames would either
    poison the mean with NaN or charge a correct rejection as a zero. Use
    `summarize_2d_scores` to get the two numbers separately.
    """
    # Imported here rather than at module scope: these pull in MONAI, which
    # is slow to import and not needed to draw ROIs.
    from medsam2_gt_eval import assd, hd95, iou as iou_metric, nsd

    meta = load_2d_case(case_dir)
    z = meta["z"]
    height, width = meta["shape_hw"]
    dy, dx = meta["spacing_mm_yx"]
    px_mm2 = dy * dx
    # The masks are (H, W) = (Y, X), so the spacing pair is already in order.
    spacing = (dy, dx)

    rows = []
    for frame in meta["frames"]:
        if not frame["annotated"]:
            continue

        t = frame["source_frame_index"]
        gt = points_to_mask([tuple(p) for p in frame["points"]], (height, width))

        result = masker.compute_frame(t)
        mask_3d = result["mask_3d"]
        if mask_3d.shape[:2][::-1] != (height, width):
            raise ValueError(
                f"frame {t}: mask_3d {mask_3d.shape} does not match the annotated "
                f"plane {(height, width)} — is this the same series the ROIs were drawn on?"
            )
        pred = mask_3d[:, :, z].T.astype(bool)

        bbox = result["bbox"]
        # Outside the tracked bbox's z range compute_3d_mask never writes, so an
        # empty prediction there is structural rather than the model deciding
        # the lesion is absent. Recorded so a correct rejection isn't over-read.
        z_in_bbox = bool(int(bbox.z_min) <= z < int(bbox.z_max))

        gt_empty, pred_empty = not gt.any(), not pred.any()

        if not gt_empty:
            outcome = "segmented"
        elif pred_empty:
            outcome = "correct rejection"
        else:
            outcome = "false positive"

        # Surface metrics need two non-empty surfaces. On an empty-GT frame
        # they are undefined, and MONAI would return inf/NaN anyway -- those
        # frames are scored as detection (specificity), not geometry.
        if gt_empty:
            geometry = {m: float("nan") for m in METRICS[1:]}
            geometry["iou"] = float("nan")
        else:
            geometry = {
                f"nsd@{NSD_TOLERANCE_MM}mm": nsd(pred, gt, spacing, NSD_TOLERANCE_MM),
                "assd_mm": assd(pred, gt, spacing),
                "hd95_mm": hd95(pred, gt, spacing),
                "iou": iou_metric(pred, gt),
            }

        rows.append({
            "source_frame_index": t,
            "time_s": frame["time_s"],
            "phase_label": frame.get("phase_label", ""),
            "gt_area_px": int(gt.sum()),
            "gt_area_mm2": float(gt.sum() * px_mm2),
            "pred_area_px": int(pred.sum()),
            "pred_area_mm2": float(pred.sum() * px_mm2),
            "gt_empty": gt_empty,
            "pred_empty": pred_empty,
            "z_in_tracked_bbox": z_in_bbox,
            # NaN on empty GT by design — see the docstring
            "dice": float("nan") if gt_empty else dice_score(pred, gt),
            **geometry,
            "outcome": outcome,
        })
        if return_masks:
            rows[-1]["gt_mask"] = gt
            rows[-1]["pred_mask"] = pred

        if verbose:
            score = "  —  " if gt_empty else f"{rows[-1]['dice']:.3f}"
            print(f"  frame {t:>4}  {frame['time_s']:>7.1f}s  dice {score}  {outcome}")

    return rows


def summarize_2d_scores(rows):
    """
    Split summary: segmentation quality on frames where the lesion is present,
    detection specificity on frames where it is not.

    These are different questions measured on disjoint frames and must not be
    combined into a single headline number.
    """
    present = [r for r in rows if not r["gt_empty"]]
    absent = [r for r in rows if r["gt_empty"]]
    rejected = [r for r in absent if r["pred_empty"]]
    false_pos = [r for r in absent if not r["pred_empty"]]

    return {
        "n_frames": len(rows),
        # segmentation, lesion present
        "n_lesion_present": len(present),
        "mean_dice": float(np.mean([r["dice"] for r in present])) if present else float("nan"),
        "median_dice": float(np.median([r["dice"] for r in present])) if present else float("nan"),
        "mean_iou": float(np.mean([r["iou"] for r in present])) if present else float("nan"),
        # detection, lesion absent
        "n_lesion_absent": len(absent),
        "n_correct_rejection": len(rejected),
        "specificity": (len(rejected) / len(absent)) if absent else float("nan"),
        "mean_false_positive_mm2": (
            float(np.mean([r["pred_area_mm2"] for r in false_pos])) if false_pos else 0.0
        ),
        # how many rejections were structural (z outside the tracked bbox)
        "n_rejection_outside_bbox": sum(1 for r in rejected if not r["z_in_tracked_bbox"]),
    }


def plot_2d_overlay(case_dir, rows, modality="bmode", enhance=True, ncols=5,
                    panel_size=3.0, frames=None):
    """
    Ground truth and MedSAM2 prediction drawn on the same image, one panel per
    annotated frame.

    `rows` must come from `evaluate_2d_case(..., return_masks=True)`. Images are
    read back from the saved `.npz` rather than re-sliced from the source
    series, so the panels show exactly the pixels the ROIs were drawn on.

    Contour colours follow the convention the other viewers already use
    (`Medsam2SliceViewerBase._render`): **red solid = reference/ground truth**,
    **lime dashed = MedSAM2 prediction**. Panel borders flag the frames where
    the lesion is absent -- green for a correct rejection, red for a false
    positive -- since those carry no Dice and would otherwise look like blank
    panels with nothing to say.
    """
    meta = load_2d_case(case_dir)
    npz_path = os.path.join(case_dir, f"{meta['case_id']}.npz")
    with np.load(npz_path) as data:
        images = data["imgs" if modality == "bmode" else "imgs_ceus"]
        saved_frames = data["frame_indices"].tolist()

    rows = [r for r in rows if frames is None or r["source_frame_index"] in frames]
    if not rows:
        raise ValueError("No frames to plot.")
    if "gt_mask" not in rows[0]:
        raise ValueError(
            "These rows carry no masks. Re-run just the evaluate line — the "
            "masker keeps its per-frame cache, so there is no need to rebuild "
            "it:\n\n"
            "    rows = evaluate_2d_case(masker, case_dir)\n"
        )

    ncols = min(ncols, len(rows))
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size * ncols, panel_size * 1.12 * nrows),
                             squeeze=False)

    for ax, row in zip(axes.ravel(), rows):
        image = images[saved_frames.index(row["source_frame_index"])]
        ax.imshow(enhance_bmode_noise(image) if enhance else slice_to_rgb(image)[..., 0],
                  cmap="gray")

        # contour() on an all-false mask draws nothing but still warns, so both
        # are guarded -- an empty GT or an empty prediction is a normal state here.
        # The prediction is drawn thinner than the GT on purpose: on a good
        # frame the two contours coincide, and at equal weight the lime one
        # hides the red entirely, making a near-perfect result look like the
        # GT is missing. A thicker red underneath shows through the dashes.
        if row["gt_mask"].any():
            ax.contour(get_mask_boundary(row["gt_mask"]), colors="red", linewidths=2.4)
        if row["pred_mask"].any():
            ax.contour(get_mask_boundary(row["pred_mask"]), colors="lime",
                       linewidths=1.2, linestyles="dashed")

        border = {"segmented": "0.6", "correct rejection": "C2",
                  "false positive": "C3"}[row["outcome"]]
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2.5 if row["gt_empty"] else 1.0)
        ax.set_xticks([]); ax.set_yticks([])

        headline = (f"Dice {row['dice']:.3f}" if not row["gt_empty"]
                    else row["outcome"])
        ax.set_title(f"f{row['source_frame_index']} · {row['time_s']:.0f}s\n{headline}",
                     fontsize=9)

    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")

    fig.legend(handles=[
        plt.Line2D([], [], color="red", lw=1.2, label="ground truth"),
        plt.Line2D([], [], color="lime", lw=1.2, ls="--", label="MedSAM2 prediction"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=9)

    fig.suptitle(
        f"{meta['subject_id']} ({meta['standardized_subject_id']}) — "
        f"axial z={meta['z']}, {modality}", fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation over every annotated case
# ─────────────────────────────────────────────────────────────────────────────

def discover_2d_cases(root_2d):
    """Every annotated case directory under `{root_2d}/processed/`, sorted."""
    pattern = os.path.join(root_2d, "processed", "subject_*", "sequence_*")
    return [d for d in sorted(glob.glob(pattern))
            if glob.glob(os.path.join(d, "*.json"))]


class _LazyBmode:
    """
    Stand-in for a loaded `UltrasoundImage`, exposing only what the maskers
    read: `pixel_data`, `pixdim`, `frame_rate`, `scan_name`.

    `src.image_loading.nifti.EntryClass` does
    `np.asarray(img.dataobj, dtype=np.uint8)`, materialising the whole 4D
    series -- 7.4 GB for one of these B-mode files. Batch evaluation touches
    fifteen of them, and every masker only ever reads one frame at a time via
    `pixel_data[:, :, :, frame_idx]`, so the nibabel proxy is kept lazy and
    each frame is read off disk on demand (~0.1 s, verified bit-identical to
    the eager loader's frame).
    """

    def __init__(self, path):
        img = nib.load(path)
        zooms = img.header.get_zooms()
        self.pixel_data = img.dataobj
        self.pixdim = tuple(float(z) for z in zooms[:3])
        self.frame_rate = (1.0 / float(zooms[3])) if len(zooms) > 3 and zooms[3] > 0 else 1.0
        self.scan_name = os.path.basename(path)


def load_case_inputs(meta):
    """
    (seg_data, bmode_image_data) for one annotated 2D case.

    The reference VOI is re-globbed from the B-mode file's own directory
    rather than read from the case metadata, so a renamed or moved source is
    still found by `find_voi_file`'s visit+bolus matching.
    """
    from src.entrypoints import seg_loading_step   # needs the ceus engine on sys.path

    voi_path = find_voi_file(os.path.dirname(meta["bmode_path"]),
                             meta["visit"], meta["bolus"])
    # seg_loading_step only dispatches, and the nifti seg loader ignores its
    # image_data argument entirely -- so the 4D CEUS series never has to be
    # loaded just to read a VOI and its motion-compensation extension.
    seg_data = seg_loading_step("nifti", None, voi_path, meta["ceus_path"])
    return seg_data, _LazyBmode(meta["bmode_path"])


def evaluate_2d_dataset(root_2d, variants, model_cfg, device="cuda",
                        cases=None, verbose=True, **common_kwargs):
    """
    Score every annotated 2D case under one or more *variants*, on exactly the
    same frames, so the comparison between them is paired.

    A variant is whatever you want to vary -- the reconstruction method, the
    checkpoint, a masker option, or any combination:

        variants = {
            "adaptive-bbox": {"masker_cls": Medsam2AdaptiveBboxMasker,
                              "checkpoint": CKPT},
            "axial-prop":    {"masker_cls": Medsam2AxialPropagationViewer,
                              "checkpoint": CKPT},
            "synthetic-cor": {"checkpoint": CKPT, "synthetic_coronal": True},
        }

    Each value is a dict of constructor kwargs and must carry `checkpoint`;
    `masker_cls` defaults to `Medsam2AdaptiveBboxMasker`. A bare string is
    shorthand for `{"checkpoint": <str>}`. Anything in `common_kwargs` is
    applied to every variant, with per-variant keys winning -- which is how
    options only one class accepts (`synthetic_coronal` is on the adaptive
    masker, not the propagation viewer) stay out of the others' constructors.

    Returns the flat list of per-frame rows from `evaluate_2d_case`, each
    tagged with `variant`, `checkpoint`, `case`, `subject_id` and `z`. Masks
    are not retained -- call `evaluate_2d_case` directly on one case when you
    want overlays.
    """
    import torch
    from medsam2_3d_mask import Medsam2AdaptiveBboxMasker

    case_dirs = list(cases) if cases is not None else discover_2d_cases(root_2d)
    if not case_dirs:
        raise ValueError(f"No annotated cases found under {root_2d}/processed/")

    rows = []
    for label, spec in variants.items():
        spec = {"checkpoint": spec} if isinstance(spec, str) else dict(spec)
        kwargs = {**common_kwargs, **spec}
        checkpoint = kwargs.pop("checkpoint")
        masker_cls = kwargs.pop("masker_cls", Medsam2AdaptiveBboxMasker)
        kwargs.setdefault("smooth_sigma_mm", 0)

        if verbose:
            print(f"\n=== {label}: {masker_cls.__name__} @ {os.path.basename(checkpoint)} ===")
        for i, case_dir in enumerate(case_dirs, 1):
            meta = load_2d_case(case_dir)
            seg_data, bmode = load_case_inputs(meta)
            masker = masker_cls(seg_data, bmode, model_cfg, checkpoint,
                                device=device, **kwargs)
            try:
                case_rows = evaluate_2d_case(masker, case_dir, verbose=False,
                                             return_masks=False)
            finally:
                # One model per case keeps the cases independent; freeing it
                # here stops fifteen checkpoints' worth of weights piling up
                # on the GPU. The propagation viewer also holds a per-frame
                # full-volume mask cache, which goes with it.
                del masker
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            case_id = f"{meta['standardized_subject_id']}_{meta['standardized_sequence_id']}"
            for r in case_rows:
                r["variant"] = label
                r["checkpoint"] = checkpoint
                r["case"] = case_id
                r["subject_id"] = meta["standardized_subject_id"]
                r["z"] = meta["z"]
            rows.extend(case_rows)

            if verbose:
                present = [r["dice"] for r in case_rows if not r["gt_empty"]]
                score = f"{np.mean(present):.3f}" if present else " n/a "
                print(f"  [{i:>2}/{len(case_dirs)}] {case_id:<34} mean Dice {score}"
                      f"  ({len(present)} present, {len(case_rows) - len(present)} empty)")
    return rows


def summarize_metrics(rows, metrics=METRICS, npz_train_dir=None):
    """
    Mean +/- SD for each metric, at three levels of aggregation, per variant
    and tier. Returns long-form rows (one per variant x tier x metric), which
    is the shape both the table and the figure want.

    All three levels are reported because they answer different questions and
    disagree on this data:

      per frame   -- every annotated frame equally. Over-precise: the frames
                     come from 9 subjects, and frames within a case are the
                     same lesion seconds apart, so the SD understates the real
                     spread.
      per case    -- each (subject, sequence, z) contributes one mean.
      per subject -- each patient contributes one mean however many sequences
                     they have. subject_000009 alone supplies 6 of the 15
                     cases, so this is the only level at which a single
                     patient cannot dominate. Quote this one.

    `npz_train_dir` splits rows into `holdout` and `train` tiers by subject --
    by subject, never by case. The tier describes what the *checkpoint* saw,
    so it only means something when the variants share a checkpoint (as when
    comparing reconstruction methods) or were trained on the same subjects.

    Frames with an empty GT carry no Dice and no surface metric -- they are
    geometry-undefined, not zero -- and are summarised separately as
    specificity, repeated on each row of a variant/tier for convenience.
    """
    if npz_train_dir:
        trained = {os.path.basename(p)[: -len(".npz")].split("_sequence_")[0]
                   for p in glob.glob(os.path.join(npz_train_dir, "*.npz"))}
    else:
        trained = None

    def stats(values):
        values = [v for v in values if v is not None and not np.isnan(v)]
        if not values:
            return float("nan"), float("nan"), 0
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return float(np.mean(values)), sd, len(values)

    out = []
    for variant in sorted({r["variant"] for r in rows}):
        for tier in ("all", "holdout", "train"):
            if tier != "all" and trained is None:
                continue
            sel = [r for r in rows if r["variant"] == variant]
            if tier != "all":
                want = tier == "train"
                sel = [r for r in sel if (r["subject_id"] in trained) == want]
            if not sel:
                continue

            present = [r for r in sel if not r["gt_empty"]]
            absent = [r for r in sel if r["gt_empty"]]
            specificity = ((sum(1 for r in absent if r["pred_empty"]) / len(absent))
                           if absent else float("nan"))
            # Lesion present but nothing predicted. Dice scores these 0, but a
            # surface distance has no surface to measure from and comes back
            # inf -> NaN, so they drop out of ASSD/HD95 entirely. That makes
            # the distance metrics an average over an easier subset than Dice,
            # and the count has to travel with them or the two look comparable
            # when they are not.
            n_miss = sum(1 for r in present if r["pred_empty"])

            for metric in metrics:
                by_case, by_subject = {}, {}
                for r in present:
                    by_case.setdefault(r["case"], []).append(r[metric])
                    by_subject.setdefault(r["subject_id"], []).append(r[metric])

                f_mean, f_sd, n_f = stats([r[metric] for r in present])
                c_mean, c_sd, n_c = stats([float(np.nanmean(v)) for v in by_case.values()])
                s_mean, s_sd, n_s = stats([float(np.nanmean(v)) for v in by_subject.values()])

                out.append({
                    "variant": variant, "tier": tier, "metric": metric,
                    "n_subjects": n_s, "n_cases": n_c, "n_frames": n_f,
                    "mean_per_frame": f_mean, "sd_per_frame": f_sd,
                    "mean_per_case": c_mean, "sd_per_case": c_sd,
                    "mean_per_subject": s_mean, "sd_per_subject": s_sd,
                    "n_absent": len(absent), "specificity": specificity,
                    "n_missed": n_miss,
                })
    return out


def per_subject_table(rows, metric, npz_train_dir=None):
    """
    {subject_id: {variant: mean metric}} plus a `tier` column, as a dict of
    dicts -- one row per subject, aggregated first within subject so that a
    patient with six sequences still counts once.
    """
    trained = set()
    if npz_train_dir:
        trained = {os.path.basename(p)[: -len(".npz")].split("_sequence_")[0]
                   for p in glob.glob(os.path.join(npz_train_dir, "*.npz"))}

    table = {}
    for r in rows:
        if r["gt_empty"] or np.isnan(r[metric]):
            continue
        table.setdefault(r["subject_id"], {}).setdefault(r["variant"], []).append(r[metric])

    out = {}
    for subject, per_variant in sorted(table.items()):
        out[subject] = {v: float(np.mean(vals)) for v, vals in per_variant.items()}
        out[subject]["tier"] = "train" if subject in trained else "holdout"
    return out


def plot_variant_comparison(rows, metrics=METRICS, npz_train_dir=None,
                            figsize=(11, 8)):
    """
    One paired slope panel per metric: each subject is a line joining its mean
    under one variant to its mean under the other.

    Paired lines rather than two bars or box plots, because both variants ran
    on identical frames -- so what matters is whether each *subject* moved and
    in which direction, which a pair of summary bars hides entirely. With only
    9 subjects, the direction and consistency of the lines carry more than any
    error bar does.

    Colour separates holdout from train subjects; the black marker is the mean
    +/- SD across subjects. Panel titles carry the arrow for which direction is
    better, since Dice and NSD go up while ASSD and HD95 go down.
    """
    variants = sorted({r["variant"] for r in rows})
    if len(variants) != 2:
        raise ValueError(f"Expected exactly 2 variants to pair, found {variants}")
    left, right = variants

    n = len(metrics)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for ax, metric in zip(axes.ravel(), metrics):
        table = per_subject_table(rows, metric, npz_train_dir)
        deltas = []
        for subject, vals in table.items():
            if left not in vals or right not in vals:
                continue
            colour = "C0" if vals["tier"] == "train" else "C3"
            ax.plot([0, 1], [vals[left], vals[right]], "-o", color=colour,
                    alpha=0.65, ms=4, lw=1.2)
            deltas.append(vals[right] - vals[left])

        # Offset outside the paired lines rather than on top of them -- at
        # x=0/1 the mean marker lands exactly where every subject's endpoint
        # already is and neither reads clearly.
        for x, variant in ((-0.22, left), (1.22, right)):
            vals = [v[variant] for v in table.values() if variant in v]
            if vals:
                ax.errorbar(x, np.mean(vals), yerr=np.std(vals, ddof=1) if len(vals) > 1 else 0,
                            fmt="s", color="black", ms=7, capsize=5, zorder=5)

        higher = METRIC_HIGHER_IS_BETTER.get(metric, True)
        better = sum(1 for d in deltas if (d > 0) == higher)
        arrow = "higher is better" if higher else "lower is better"
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}  ({arrow})", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels([left, right])
        ax.set_xlim(-0.45, 1.45)
        ax.grid(axis="y", alpha=0.25)
        if deltas:
            sd = np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0
            ax.set_xlabel(f"Δ {np.mean(deltas):+.3f} ± {sd:.3f}   "
                          f"({better}/{len(deltas)} favour {right})", fontsize=8)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.legend(handles=[
        plt.Line2D([], [], color="C3", marker="o", ls="-", label="holdout subject"),
        plt.Line2D([], [], color="C0", marker="o", ls="-", label="train subject"),
        plt.Line2D([], [], color="black", marker="s", ls="none", label="mean ± SD"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{left} vs {right} — per-subject means over "
                 f"{len({r['case'] for r in rows})} annotated cases", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig
