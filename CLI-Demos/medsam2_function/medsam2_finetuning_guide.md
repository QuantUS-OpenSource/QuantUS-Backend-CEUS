# Fine-tuning MedSAM2 on 3DMPUS liver tumor data

This walks through the full pipeline for fine-tuning MedSAM2 on our own
segmented CEUS/B-mode volumes: from a labeled `segmentations/` folder to a
new `.pt` checkpoint you can drop into the inference notebook.

Scripts referenced live in [`CLI-Demos/`](..) (`build_npz_dataset.py`) and
the `MedSAM2/` checkout (sibling of `engines/` in the repo root) — the
`training/` package, its Hydra configs under `sam2/configs/`, and the launch
scripts at its root.

## How fine-tuning actually works here

MedSAM2 treats a 3D volume as a "video": the Z (axial slice) stack plays the
role that time plays in ordinary video object segmentation. During training,
the sampler grabs a random contiguous window of slices, gives the model a
box/point prompt on the first slice that contains the tumor, and the model
has to propagate the mask through the rest of the window using the same
memory-attention mechanism it uses at inference. So fine-tuning directly
drills the axial-propagation behavior we rely on — it doesn't need frame-by-
frame labels, just a 3D mask (or several) per case, most conveniently a
mask that's contiguous across a run of slices.

A checkpoint (`.pt`) is just a snapshot of the model's weights — a Python
dict `{'model': state_dict, ...}`. Fine-tuning loads an existing checkpoint
as the starting point, keeps updating those weights via backprop on our
labeled cases, and periodically writes out **new** `.pt` files. The
checkpoint you started from is never modified.

## Step 1 — Label volumes

Each case lives at:

```
{processed_root}/{subject_id}/{sequence_id}/
    image.nii.gz          # B-mode volume (X, Y, Z), uint8
    segmentations/*.nii.gz  # your ground-truth mask, same (X, Y, Z) shape
```

`extract_finetune_dataset.py` (already in this folder) builds that tree from
the raw 4D series and the motion-compensation manifest — see its own
docstring. What matters for fine-tuning is just: **a case counts as labeled
once its `segmentations/` folder has a mask file in it.** Not every case
needs to be labeled before you start; only labeled cases get used.

The mask should be binary (0 = background, 1 = tumor) and the same shape as
`image.nii.gz`. It doesn't need to be labeled on every slice — only the
slices where the tumor is actually visible — but it does need to be a
single contiguous run of slices (a real 3D VOI), not scattered single-slice
annotations, since the sampler needs a "first frame with the object" to
prompt from and continuous slices after it to propagate onto.

## Step 2 — Convert labeled cases to `.npz`

The trainer doesn't read the `image.nii.gz` / `segmentations/` layout
directly — it reads one `.npz` file per case (`training/dataset/
vos_raw_dataset.py:NPZRawDataset`), holding:

- `imgs`: `(Z, H, W)` uint8 — the axial slice stack
- `gts`: `(Z, H, W)` integer labels — 0 background, 1 tumor

Run the converter:

```bash
cd engines/ceus/CLI-Demos
python build_npz_dataset.py \
    --processed-root /path/to/MedSAM2_finetune_data/processed \
    --output-dir /path/to/MedSAM2_finetune_data/npz_train
```

It scans every `{subject_id}/{sequence_id}/`, skips any case whose
`segmentations/` folder is empty, and writes `{subject_id}_{sequence_id}.npz`
for the rest. Re-run it any time you label more cases — it's safe to
overwrite the output directory.

**Held-out cases:** before pointing the config at this folder, pull a couple
of cases out into a separate `npz_holdout/` folder so you have something
fine-tuning never saw, to sanity-check the result on. Split by **subject**,
not sequence — several subjects have multiple labeled sequences, and if you
hold out one sequence from a subject while training on another sequence
from the *same* subject, the holdout check is contaminated (the model may
have effectively already seen that patient's anatomy).

## Step 3 — Configure the YAML

Copy an existing fine-tune config as your starting point rather than writing
one from scratch:

```bash
cd MedSAM2
cp sam2/configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml \
   sam2/configs/sam2.1_hiera_tiny512_3DMPUS.yaml
```

Then edit these fields (all inside the one file):

**1. Dataset path — the one that actually matters:**

```yaml
trainer:
  data:
    train:
      datasets:
        - dataset:
            datasets:
              - video_dataset:
                  folder: /path/to/MedSAM2_finetune_data/npz_train   # <-- set this, must be absolute
```

⚠️ **Gotcha:** there's *also* a `dataset.folder:` key near the top of the
file (under `scratch:`). It looks like the obvious place to set the path,
and the `train.py --dataset-path` CLI flag even claims to override it — but
nothing in the actual training graph reads that top-level key back. The
path the trainer really uses is the nested `video_dataset.folder` shown
above. Set both if you like (harmless), but the nested one is the one that
counts.

**2. Which checkpoint to start from:**

```yaml
trainer:
  checkpoint:
    model_weight_initializer:
      state_dict:
        checkpoint_path: checkpoints/MedSAM2_MRI_LiverLesion.pt   # relative to MedSAM2/
```

This is the "starting weights" for fine-tuning. We use the existing MedSAM2
liver-lesion checkpoint (the one already used for inference) rather than
the base `sam2.1_hiera_tiny.pt` (natural-image SAM2, no medical adaptation
at all), since it's already close to our task. It's worth trying
`MedSAM2_US_Heart.pt` too at some point — it's ultrasound-domain-adapted
(matching our imaging modality) even though it's a different anatomy, which
is a different kind of head start than the liver-lesion checkpoint (right
anatomy, wrong modality — MRI). Whichever you use, the loader expects the
checkpoint file to be a dict with a `'model'` key, which is the format all
the `checkpoints/*.pt` files (and the ones fine-tuning produces) already use.

**3. GPU count** — match your machine:

```yaml
launcher:
  gpus_per_node: 1   # or however many GPUs you actually have
```

(Can also be passed as `--num-gpus` on the command line instead.)

**4. Object count** — how many distinct labeled classes per case:

```yaml
scratch:
  max_num_objects: 1   # we only ever label one tumor per case
```

**5. Epoch count** — how many passes over the training set:

```yaml
scratch:
  num_epochs: 50   # tune this; see "How many epochs" below
```

Other knobs worth knowing about but usually fine at their defaults:

- `scratch.train_video_batch_size` (2) / `scratch.num_frames` (8) — how many
  clips per batch, how many slices per clip. Raise the batch size if you
  have GPU memory to spare and enough training videos to fill bigger
  batches meaningfully.
- `trainer.checkpoint.save_freq` (10) — write a named
  `checkpoint_{epoch}.pt` snapshot every N epochs, on top of the
  always-current rolling `checkpoint.pt`.
- `trainer.optim.optimizer` / the LR scheduler block — AdamW with cosine
  decay and a lower learning rate specifically for the image encoder
  (`scratch.vision_lr`). Leave as-is unless you have a specific reason to
  touch it.

### How many epochs?

There's no validation loop wired into this config (`trainer.mode:
train_only`), so there's no automatic "best epoch" to pick — the loss curve
is your only training-time signal, and with a small number of cases it's
noisy (a single hard clip can dominate one epoch's average, since each
epoch is only a handful of steps). Don't chase a perfectly smooth loss
curve down to zero; that's a sign of overfitting the training clips, not
generalizing. When in doubt, evaluate a few epoch checkpoints
(`checkpoint_10.pt`, `checkpoint_20.pt`, ...) against the held-out cases
from Step 2 and pick by eye, rather than trusting the training loss alone.

## Step 4 — Run it

```bash
cd MedSAM2
python training/train.py \
    -c configs/sam2.1_hiera_tiny512_3DMPUS.yaml \
    --output-path ./exp_log/3DMPUS_liver_tumor \
    --use-cluster 0 --num-gpus 1 --num-nodes 1
```

(`single_node_train_medsam2.sh` at the repo root is the same command
wrapped in a shell script — copy and adjust it if you'd rather have a saved
launch script than type the command each time.)

`--output-path` is where everything for this run goes:

```
exp_log/3DMPUS_liver_tumor/
    config.yaml, config_resolved.yaml   # exact config used, for reproducibility
    logs/log.txt                        # human-readable training log
    tensorboard/                        # TensorBoard event file
    checkpoints/
        checkpoint.pt                   # always the latest epoch
        checkpoint_10.pt, checkpoint_20.pt, ...   # snapshots every save_freq epochs
```

With a small dataset (a dozen or so cases) on a modern GPU, expect this to
take minutes, not hours — 50 epochs over 14 cases at batch size 2 is only a
few hundred optimizer steps total.

## Step 5 — Monitor

- `tail -f exp_log/3DMPUS_liver_tumor/logs/log.txt` — per-step and per-epoch
  loss breakdown (mask / dice / iou / class) plus an ETA. The most reliable
  option, especially for short runs.
- `tensorboard --logdir exp_log/3DMPUS_liver_tumor/tensorboard` — loss
  curves, per-parameter-group LR schedule, batch/data timing, viewed in a
  browser.
- `watch -n 2 nvidia-smi` — confirm the GPU is actually being exercised.

## Step 6 — Evaluate

Fine-tuning checkpoints are saved in the same `{'model': state_dict}` format
as every other `checkpoints/*.pt` file, so they're a drop-in swap:

```python
checkpoint = os.path.join(medsam2_path, 'exp_log', '3DMPUS_liver_tumor', 'checkpoints', 'checkpoint.pt')
```

in [`MedSam2_inference.ipynb`](../MedSam2_inference.ipynb). Load your
held-out cases (the ones set aside in Step 2, never seen during training)
through the 3D viewer and compare masks against the original
`MedSAM2_MRI_LiverLesion.pt` — this qualitative check matters more than the
training loss curve, since the loss on a dataset this size is too noisy to
read confidently on its own.

## Iterating

As more cases get labeled: re-run `build_npz_dataset.py` (Step 2) to pick
them up, then re-run training (Step 4) — either from scratch again off
`MedSAM2_MRI_LiverLesion.pt`, or continuing from your last fine-tuned
`checkpoint.pt` by pointing `checkpoint_path` at it instead.
