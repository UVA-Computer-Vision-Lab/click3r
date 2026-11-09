# Bundled DUSt3R

The parts of DUSt3R and CroCo that Click3R needs. Install the dependencies from the root
`requirements.txt`.

Upstream: [naver/dust3r](https://github.com/naver/dust3r) at commit
`4c24a6ebf04809f2cfe59915e51779c8984aaa40`, with
[naver/croco](https://github.com/naver/croco) at commit
`d7de0705845239092414480bd829228723bf20de` in `croco/`.
See [dataset preparation](../../docs/datasets.md) for training data instructions.

Keep `LICENSE`, `NOTICE`, and `croco/LICENSE` / `croco/NOTICE` in distributions.

## Modifications

The Click3R authors modified these files relative to the upstream commits:

| File | Change |
|---|---|
| `dust3r/datasets/{arkitscenes,blendedmvs,co3d,habitat,megadepth,scannetpp,staticthings3d,waymo}.py` | Views also return `img_path` (and `instance` where missing) so batches from all datasets collate |
| `dust3r/datasets/habitat.py` | Detects which views of a viewpoint exist on disk and samples among them |
| `dust3r/datasets/utils/cropping.py` | Passes integer output sizes to the resize calls |
| `dust3r/inference.py` | Skips the `sparse_prompts` dictionary and `img_path` strings when moving a batch to the device |
| `dust3r/losses.py` | Adds `PromptAwareConfidenceLoss` (not used by the Click3R recipe) |
| `dust3r/model.py`, `croco/utils/misc.py` | Load checkpoints with `weights_only=False` |
| `dust3r/training.py` | Optional Weights & Biases logging |
| `datasets_preprocess/preprocess_co3d.py` | `prepare_sequences()` can read another set list and convert a given list of sequences; `scripts/table1/prepare_data.py` uses this for the Table 1 CO3D data |
| `datasets_preprocess/preprocess_scannetpp.py` | Drops selected frames without a COLMAP pose (remapping the pairs) and accepts RGB anonymization masks; these changes built the Click3R ScanNet++ data |
| `dust3r/demo.py` | `demo.launch(share=True)`; the Click3R demo does not use this launcher |
| `README.md`, `croco/README.MD` | Shortened to describe the bundled subset |
