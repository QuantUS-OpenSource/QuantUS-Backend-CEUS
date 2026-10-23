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

Two things that bite when preparing masks:

- **One mask file per `segmentations/` folder.** The converter stops with an
  error if it finds two (e.g. an old ITK-SNAP `Tumor_Segmentation.nii.gz` next
  to a new QuantUS `Tumor_segmentation.nii.gz` — the names differ only in
  case). Move the stale one out of `processed/` before building.
- **A mask belongs to the exact frame that was extracted.** `image.nii.gz` is a
  single frame from the 4D series, chosen by the `frame_idx` column of
  `manifest.csv` (also recorded as `source_frame_index` in `image.json`). If you
  change a case's `frame_idx` and re-extract it, the old mask no longer matches
  the image: move it aside and redraw. Re-extracting never touches
  `segmentations/`, so a stale mask would otherwise be picked up silently. To
  re-extract only some cases, call `extract_case(row, output_root)` on those
  manifest rows instead of `extract_all` (which re-reads every ~7 GB series).

## Step 2 — Convert labeled cases to `.npz`

Choose one of two modes: a **single train/holdout split** (below), or
**k-fold cross-validation** ([Step 2b](#step-2b--k-fold-cross-validation)),
which is the better way to evaluate with only ~10 patients.

The trainer doesn't read the `image.nii.gz` / `segmentations/` layout
directly — it reads one `.npz` file per case (`training/dataset/
vos_raw_dataset.py:NPZRawDataset`), holding:

- `imgs`: `(Z, H, W)` uint8 — the axial slice stack
- `gts`: `(Z, H, W)` integer labels — 0 background, 1 tumor

Run the converter:

```bash
cd engines/ceus/CLI-Demos
python build_npz_dataset.py \
--processed-root /media/ahmed-el-kaffas/20TB-HDD/Yuanshan/3DMPUS/MedSAM2_finetune_data/processed \
--output-dir /media/ahmed-el-kaffas/20TB-HDD/Yuanshan/3DMPUS/MedSAM2_finetune_data/npz_train \
--holdout-dir /media/ahmed-el-kaffas/20TB-HDD/Yuanshan/3DMPUS/MedSAM2_finetune_data/npz_holdout \
--holdout-subjects subject_000003 subject_000006 subject_000008
```

It scans every `{subject_id}/{sequence_id}/`, skips any case whose
`segmentations/` folder is empty, and writes `{subject_id}_{sequence_id}.npz`
for the rest. Re-run it any time you label more cases — it's safe to
overwrite the output directory.

**Held-out cases:** pass `--holdout-subjects` and `--holdout-dir` so the split
is made in the same pass that writes the files:

```bash
python build_npz_dataset.py \
    --processed-root /path/to/MedSAM2_finetune_data/processed \
    --output-dir     /path/to/MedSAM2_finetune_data/npz_train \
    --holdout-dir    /path/to/MedSAM2_finetune_data/npz_holdout \
    --holdout-subjects subject_000003 subject_000006 subject_000008
```

Split by **subject**, never by sequence — most subjects have several labeled
sequences of the same lesion at a different visit or bolus, so holding out one
sequence while training on another from that patient means the model has
already seen the anatomy. Naming a subject holds out *every* sequence of that
subject; naming one that has no labels yet is fine and future-proofs the rebuild.

**Do not re-run this script with only `--output-dir` once a holdout exists.**
It walks the whole `processed/` tree, so that writes every labeled case into
the training folder and silently sweeps the held-out cases back in. With the
holdout flags the routing is applied per case and any stale copy in the other
folder is deleted, so re-running is idempotent — and re-running *with* the
flags repairs a split that was already contaminated. The chosen split is
recorded in `split.json` next to the two folders.

`medsam2_gt_eval.discover_cases` reads which `.npz` files are in `npz_train/`
to decide each case's tier, so that folder is the record of what the trainer
actually saw — which is why the split lives in the filesystem rather than in a
config flag.

## Step 2b — K-fold cross-validation

With only ~10 patients a single holdout of 2–4 patients gives a noisy score
that depends on which patients happened to be held out. K-fold trains and
tests several times so every patient is tested exactly once:

```bash
cd engines/ceus/CLI-Demos/medsam2_function
python build_npz_dataset.py \
    --processed-root /path/to/MedSAM2_finetune_data/processed \
    --kfold 5 \
    --kfold-root     /path/to/MedSAM2_finetune_data/kfold
```

This writes:

```
kfold/
    fold_1/npz_train/   fold_1/npz_holdout/
    fold_2/npz_train/   fold_2/npz_holdout/
    ...
    kfold_split.json    # which subjects/cases are in each fold
    _all/               # one converted copy of every case (folds hard-link to it)
```

- The split is **by subject**: labeled subjects are sorted and dealt
  round-robin into the folds, so 10 subjects in 5 folds gives holdout pairs
  1+6, 2+7, 3+8, 4+9, 5+10. A subject is held out in exactly one fold and none
  of its sequences are ever in that fold's training set.
- Fold sizes in *cases* are uneven (a subject can have 1–6 sequences). When
  scoring, average over all cases rather than over the 5 fold means.
- Only labeled cases are used. Change `--kfold` for a different number of folds
  (it must not exceed the number of labeled subjects).
- Re-running wipes and rebuilds every `fold_k/` folder, so label more cases and
  just run it again. `--kfold` can't be combined with `--holdout-*` or
  `--train-cases`.
- Don't use these folders and the single-split `npz_train/` interchangeably:
  each trained model is only valid for the folder it was configured with.

Train one model per fold (Steps 3–4, pointing at `fold_k/npz_train`) and
evaluate each on its own `fold_k/npz_holdout` (Step 6). Afterwards, train once
on **all** subjects for the model you actually deploy, and report the k-fold
scores as its expected performance. Tuning epochs/learning rate on the k-fold
scores makes them optimistic, so fix those settings before the final run.

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

### Step 4b — Fine-tune all 5 folds

Prerequisites: the folds are built ([Step 2b](#step-2b--k-fold-cross-validation))
and `sam2/configs/sam2.1_hiera_tiny512_3DMPUS.yaml` has the settings you want
(Step 3: starting checkpoint, GPU count, epochs). The k-fold runs reuse that
file unchanged except for the data path, so set the epochs *before* you start
and keep them identical across folds — otherwise the fold scores aren't
comparable.

**1. Generate one config per fold.** Each copy differs from the template only
in the two `folder:` lines (the top-level `dataset.folder` and the nested
`video_dataset.folder`, which is the one the trainer reads):

```bash
cd MedSAM2
for k in 1 2 3 4 5; do
  sed "s#MedSAM2_finetune_data/npz_train#MedSAM2_finetune_data/kfold/fold_${k}/npz_train#g" \
      sam2/configs/sam2.1_hiera_tiny512_3DMPUS.yaml \
      > sam2/configs/sam2.1_hiera_tiny512_3DMPUS_fold${k}.yaml
done
grep -n "folder:" sam2/configs/sam2.1_hiera_tiny512_3DMPUS_fold*.yaml
```

The `grep` must show `kfold/fold_k/npz_train` on two lines for every fold file
— check that the `k` matches the filename before going on.

**2. Train the folds, one after another.** Each fold gets its own
`--output-path` so checkpoints never mix, and nothing resumes from another
fold's weights:

```bash
for k in 1 2 3 4 5; do
  python training/train.py \
      -c configs/sam2.1_hiera_tiny512_3DMPUS_fold${k}.yaml \
      --output-path ./exp_log/kfold_${k} \
      --use-cluster 0 --num-gpus 1 --num-nodes 1
done
```

**Limiting training to ~30% of the GPU (shared machine).** Use NVIDIA MPS,
which caps the share of the GPU's compute (SMs) a process may use. Start the
daemon once, then run the same loop with the cap set in that shell:

```bash
cd MedSAM2
(   # subshell: the three variables below exist only for this run
  export CUDA_MPS_PIPE_DIRECTORY=$HOME/mps
  export CUDA_MPS_LOG_DIRECTORY=$HOME/mps_log
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30    # ~30% of the SMs per training process

  nvidia-cuda-mps-control -d                     # start a private MPS daemon

  for k in 1 2 3 4 5; do
    python training/train.py \
        -c configs/sam2.1_hiera_tiny512_3DMPUS_fold${k}.yaml \
        --output-path ./exp_log/kfold_${k} \
        --use-cluster 0 --num-gpus 1 --num-nodes 1 \
        2>&1 | tee exp_log/kfold_${k}.console.log
  done

  echo quit | nvidia-cuda-mps-control            # stop the daemon when all folds finish
)
```

- The cap is **scoped to this task only**. The variables are set inside the
  `( ... )` subshell, so they vanish when it ends and nothing else in your
  terminal is affected. The MPS daemon is private to this run (its own
  `$HOME/mps` pipe directory): only processes launched with that variable
  talk to it, so other jobs on the GPU, and anything you start afterwards, run
  as normal and uncapped.
- This caps **compute only**, not memory. Training will take roughly 3× longer
  than on an idle GPU. If memory is also tight, lower
  `scratch.train_video_batch_size` (2) or `scratch.num_frames` (8) in the
  template and regenerate the fold configs (step 1) so all folds stay identical.
- The loop runs the folds one after another, so at most ~30% is in use at any
  time. Running folds side by side in separate shells (each with the same
  exports) would add up — three at once is ~90%.
- `nvidia-smi` utilization reads 100% whenever any kernel is running, so it
  won't show the cap; check per-process load with `nvidia-smi pmon -s u`.

Each fold's checkpoints land in `exp_log/kfold_{k}/checkpoints/`
(`checkpoint.pt` = last epoch, `checkpoint_10.pt`, `checkpoint_20.pt`, ...).
Don't reuse `exp_log/3DMPUS_liver_tumor` — it holds the earlier single-split
run, which was trained on older masks and a different split. To run a single
fold (or re-run one that failed), use the same command with that `k` only; to
watch it, point `tail -f` / TensorBoard at `exp_log/kfold_{k}` (Step 5).

**3. Check each run used the right data.** Nothing else records which `.npz`
files a checkpoint saw, so confirm for every fold:

- `grep folder exp_log/kfold_${k}/config_resolved.yaml` shows
  `kfold/fold_${k}/npz_train`.
- The checkpoints are newer than the `.npz` files
  (`ls -l --time-style=long-iso kfold/fold_${k}/npz_train | head -2` vs
  `exp_log/kfold_${k}/checkpoints`). If you re-run `build_npz_dataset.py
  --kfold` later, every fold's checkpoints become stale — retrain them.
- Keep `kfold/kfold_split.json` next to the results: it is the record of which
  subjects were held out in each fold.

Then evaluate each fold on its own holdout (Step 6). The generated
`*_fold{k}.yaml` configs can be left out of version control. When all folds are
done, train the deployment model on every labeled case by pointing both
`folder:` keys at `kfold/_all` (it holds one `.npz` per labeled case) with its
own `--output-path`, e.g. `./exp_log/final_all`.

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

For k-fold, evaluate fold *k*'s checkpoint only on `kfold/fold_k/npz_holdout`
(never on another fold's holdout — that fold's training set contains those
subjects). Compare against the pretrained `MedSAM2_MRI_LiverLesion.pt` run on
the same holdout cases to see whether fine-tuning helped.

## Iterating

As more cases get labeled: re-run `build_npz_dataset.py` (Step 2) to pick
them up, then re-run training (Step 4) — either from scratch again off
`MedSAM2_MRI_LiverLesion.pt`, or continuing from your last fine-tuned
`checkpoint.pt` by pointing `checkpoint_path` at it instead.
