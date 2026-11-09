# Evaluation

Run these from the repository root. If you used the launcher, run `source .venv/bin/activate` first.

## Table 1: standard benchmarks

Table 1 evaluates ten rows on 1,000 image pairs each from CO3D, ScanNet++ and ARKitScenes. First [prepare the data](datasets.md#held-out-table-1-data).

The default runs five rows: pretrained and retrained DUSt3R, and Click3R without prompts, with `1p_best5` and with the grid oracle.

```bash
python scripts/download_checkpoints.py --dust3r-retrained-source
python evaluation/reproduce_table1.py --data-root /path/to/table1_data --output-dir outputs/table1
python evaluation/summarize_table1.py outputs/table1 --output outputs/table1_summary
```

To run all ten rows and compare them with our values, install the baselines and build the CroCo RoPE extension, which Pi3X needs:

```bash
python scripts/setup_environment.py --baselines
(cd third_party/dust3r/croco/models/curope && python setup.py build_ext --inplace)
python evaluation/reproduce_table1.py --all-methods --data-root /path/to/table1_data --output-dir outputs/table1_all
python evaluation/summarize_table1.py outputs/table1_all \
  --compare click3r/configs/table1/reference.json --output outputs/table1_summary
```

`--compare` fails if a value differs from `reference.json` by more than 0.1 mAA points or 0.005 pointmap error. Add `--max-pairs 2 --batch-size 2 --num-workers 0` for a quick test, and `--resume-from outputs/table1` to continue an interrupted run.

Protocol:
- Pairs whose two views are the same frame (16 of 1,000 on CO3D) are left out, because their translation direction is undefined.
- DUSt3R-family models take symmetrized views at 224 × 224.
- Pointmap error is the median `Regr3D_ScaleShiftInv(L21, gt_scale=True)` loss across pairs.
- Pose mAA@30 integrates the accuracy of `max(rotation_error, translation_error)` through 30 degrees, and a failed pose scores 180 degrees.
- `1p_best5` draws one ground-truth correspondence five times and keeps the one with the lowest reconstruction loss. The grid oracle tries one correspondence per populated 14-pixel cell and keeps the one with the lowest pose error.

## Table 2: Hard3DBench

Table 2 evaluates [Hard3DBench](datasets.md#hard3dbench). These commands run Click3R without prompts and with 1, 4 and 8 prompts:

```bash
# 2-view, In-Distribution
python evaluation/hard3dbench/two_view.py --split indist --model click3r \
  --checkpoint checkpoints/click3r.pth --num-prompts 0 1 4 8 \
  --data-root /path/to/processed_data --co3d-root /path/to/table1_data \
  --results-dir outputs/table2

# 2-view, Out-of-Distribution
python evaluation/hard3dbench/two_view.py --split ood --model click3r \
  --checkpoint checkpoints/click3r.pth --num-prompts 0 1 4 8 \
  --ood-root /path/to/hard3dbench_ood --results-dir outputs/table2
```

For the 5-view rows, use `evaluation/hard3dbench/five_view.py` with the same arguments.

`--model` also takes `dust3r_pretrained` and `dust3r_retrained` (with `--checkpoint` and `--num-prompts 0`), and `mast3r_512`, `vggt_224`, `vggt_518`, `pi3_224` and `pi3_518` (no `--checkpoint`, after `python scripts/setup_environment.py --baselines`). The baselines load the weights pinned in `click3r/configs/baseline_checkpoints.json`. Table 1 uses Pi3X and Table 2 uses Pi3 (`yyfz233/Pi3`).

Each run writes the errors of every sample to `<results-dir>/<subset>_<split>/<model>[_k<K>].json`. Rerunning a command skips finished files, so use a separate `--results-dir` with `--limit N` for a quick test.

Protocol:
- Click3R and DUSt3R take 224 × 224 inputs. In-Distribution views are cropped as in the DUSt3R loaders, Out-of-Distribution images as in DUSt3R's `load_images`.
- With K prompts, Click3R gets up to K ground-truth correspondences per pair: min(n, 10K) of the n valid view-1 pixels are drawn with seed 42, and one is kept if it reprojects into view 2 in front of the camera and agrees with the view-2 depth within 15%. A pair without a valid prompt keeps its prompt-free prediction.
- 2-view cameras come from PnP-RANSAC on the predicted pointmap, over pixels with above-mean confidence, using ground-truth intrinsics. MASt3R does the same. VGGT and Pi3 predict their cameras, and the 2-view VGGT pointmaps come from its depth head.
- 5-view: Click3R, DUSt3R and MASt3R predict the 20 ordered pairs of each tuple, and DUSt3R's global alignment recovers the five cameras. VGGT and Pi3 predict the five views in one forward pass.
- mAA@30 is as in Table 1, but over finite poses only. 5-view mAA@30 pools the ten pairwise poses of every tuple.
- Pointmap error is computed in the first camera's frame, normalized as in `Regr3D_ScaleShiftInv(L21, gt_scale=True)`. A cell reports the median.

`click3r/configs/hard3dbench/reference.json` holds our values, from A100 GPUs (the Out-of-Distribution baselines from an RTX A6000). Values differ slightly on other GPUs.

## Table 7: failure categories

Every 2-view pair has one or more failure categories: large viewpoint (relative rotation > 60°), moderate viewpoint (30–60°), low overlap (< 20% of view-1 pixels visible in view 2), large baseline (camera distance / median depth > 0.3), extreme depth range (max / min depth > 50), or other. The labels are in `click3r/configs/hard3dbench/categories.json`.

```bash
# recompute the labels
python evaluation/hard3dbench/labels.py --data-root /path/to/processed_data \
  --co3d-root /path/to/table1_data --ood-root /path/to/hard3dbench_ood \
  --output outputs/categories.json

# Table 7, from the Table 2 results
python evaluation/hard3dbench/labels.py --table7 outputs/table2
```

Unlike Table 2, a failed pose counts as a miss here.
