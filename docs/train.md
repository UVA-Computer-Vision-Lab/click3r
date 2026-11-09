# Training

[Prepare the nine datasets](datasets.md) under `/path/to/processed_data`, then run:

```bash
python train.py --data-root /path/to/processed_data --dry-run  # print the recipe
python train.py --data-root /path/to/processed_data
```

This trains for 100 epochs on 4 GPUs with 16 pairs per GPU and writes to `outputs/training`. On A100s a step takes 0.60 s, about 234 hours in total, with a peak memory of about 29,000 MiB per GPU. Run the same command again to resume.

Use `--gpus`, `--batch-size` and `--accum-iter` for other setups, and add `--skip-setup` if you installed with conda.

## Hyperparameters

| Setting | Value |
|---|---|
| Training data | Nine datasets, 100,000 pairs each per epoch |
| Validation data | Six datasets, 1,000 pairs each |
| Optimizer | AdamW, β=(0.9, 0.95), weight decay 0.05 |
| Learning rate | 1e-4 peak, 1e-6 minimum, 10-epoch warmup, then cosine |
| Mixed precision | CUDA AMP, float16 |
| Loss | DUSt3R's confidence-weighted, scale-normalized L21 regression, confidence coefficient 0.2 |
| Initialization | CroCo v2 |
| Inputs | 224 × 224, color jitter |
| Prompts per pair | 30% none, 40% uniformly 1–4, 30% uniformly 5–12 |
| Prompt token coordinate | pixel / patch size (offset 0.0) |

`--dry-run` prints the full recipe, including each dataset's split and crop.

We keep a checkpoint as best whenever any of the six validation sets reaches a new lowest median no-prompt loss. The released Click3R checkpoint is the best one, from epoch 89.

## Export a checkpoint

```bash
python scripts/export_checkpoint.py outputs/training/checkpoint-best.pth artifacts/click3r.pth \
  --training-config click3r/configs/train_click3r.json
```

This keeps only the model weights and settings.

## DUSt3R (retrained)

The baseline uses Click3R's backbone without the adapters, with the same nine training datasets, optimizer, learning-rate schedule, batch layout and CroCo v2 initialization. We trained it with the bundled DUSt3R trainer (`third_party/dust3r/dust3r/training.py`), started through upstream DUSt3R's `train.py`. The released file is the best checkpoint, from epoch 76.
