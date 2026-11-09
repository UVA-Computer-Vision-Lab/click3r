# Dataset preparation

We train Click3R on nine datasets converted to the DUSt3R format. Table 1 and the Hard3DBench CO3D samples use separate held-out splits ([Held-out Table 1 data](#held-out-table-1-data)).

| Dataset | Terms | Access |
| --- | --- | --- |
| [BlendedMVS](#blendedmvs) | [CC BY 4.0](https://github.com/YoYo000/BlendedMVS#license) | open |
| [MegaDepth](#megadepth) | [CC BY 4.0; images under their owners' licenses](https://www.cs.cornell.edu/projects/megadepth/) | open |
| [CO3D v2](#co3d-v2) | [CC BY-NC 4.0](https://github.com/facebookresearch/co3d/blob/main/LICENSE) | open |
| [StaticThings3D](#staticthings3d) | [research use only](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) | open |
| [Waymo Open Dataset](#waymo) | [non-commercial license](https://waymo.com/open/terms/) | accept license |
| [ARKitScenes](#arkitscenes) | [ARKitScenes license](https://github.com/apple/ARKitScenes/blob/main/LICENSE) | open |
| [WildRGB-D](#wildrgb-d) | [MIT file in the code repository](https://github.com/wildrgbd/wildrgbd/blob/main/LICENSE) | open |
| [Habitat](#habitat) | [HM3D EULA](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data); [ScanNet Terms of Use](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf) | request access |
| [ScanNet++](#scannet) | [ScanNet++ Terms of Use](https://scannetpp.mlsg.cit.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf) | signed agreement |

Put every converted dataset under one data root, `/path/to/processed_data`. After converting, you can check that the index and split files match ours:

```bash
python scripts/audit_datasets.py --data-root /path/to/processed_data --output audit.json
```

## Datasets

### BlendedMVS

Download the three low-res subsets from the [GitHub releases](https://github.com/YoYo000/BlendedMVS/releases) (about 205 GB) and Naver's pair file:

```bash
RAW=/path/to/blendedmvs; URL=https://github.com/YoYo000/BlendedMVS/releases/download
for i in $(seq -w 1 15); do wget -c -P $RAW $URL/v1.0.0/BlendedMVS.z$i; done
for i in $(seq -w 1 42); do wget -c -P $RAW $URL/v1.0.1/BlendedMVS1.z$i $URL/v1.0.2/BlendedMVS2.z$i; done
wget -c -P $RAW $URL/v1.0.0/BlendedMVS.zip $URL/v1.0.1/BlendedMVS1.zip $URL/v1.0.2/BlendedMVS2.zip
for s in BlendedMVS BlendedMVS1 BlendedMVS2; do
  zip -s 0 $RAW/$s.zip --out $RAW/${s}_joined.zip && rm $RAW/$s.z[0-9][0-9] $RAW/$s.zip &&
  unzip -q $RAW/${s}_joined.zip -d $RAW && rm $RAW/${s}_joined.zip || break
done
mv $RAW/BlendedMVS/* $RAW/ && rmdir $RAW/BlendedMVS
wget -P $RAW https://download.europe.naverlabs.com/ComputerVision/DUSt3R/blendedmvs_pairs.npy
sha256sum $RAW/blendedmvs_pairs.npy  # 78802fd84b9cd05d253cc9beec69581acf9cb81ce8b4c2350557d23b913e0026
```

Convert (about 85 GB of output):

```bash
python third_party/dust3r/datasets_preprocess/preprocess_blendedMVS.py \
  --blendedmvs_dir /path/to/blendedmvs --precomputed_pairs /path/to/blendedmvs/blendedmvs_pairs.npy \
  --output_dir /path/to/processed_data/blendedmvs_processed_for_dust3r
cp /path/to/blendedmvs/blendedmvs_pairs.npy /path/to/processed_data/blendedmvs_processed_for_dust3r/
```

### MegaDepth

Download MegaDepth v1 from the [project page](https://www.cs.cornell.edu/projects/megadepth/) (about 200 GB, no registration) and the pair file. The SfM archive is about 670 GB, so the command streams it and keeps only `cameras.txt` and `images.txt`:

```bash
wget -P /path/to/megadepth https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz
tar -xzf /path/to/megadepth/MegaDepth_v1.tar.gz -C /path/to/megadepth
curl -L https://www.cs.cornell.edu/projects/megadepth/dataset/MegaDepth_SfM/MegaDepth_SfM_v1.tar.xz \
  | tar -xJf - -C /path/to/megadepth/phoenix/S6/zl548/MegaDepth_v1 --strip-components=2 \
      --wildcards '*/sparse/manhattan/*/cameras.txt' '*/sparse/manhattan/*/images.txt'
wget -P /path/to https://download.europe.naverlabs.com/ComputerVision/DUSt3R/megadepth_pairs.npz
sha256sum /path/to/megadepth_pairs.npz  # 32c9d610d4d5ef3756869d2f8b6f084a7eebcc9664545e6eb8b4f70be49c5e64
```

Convert (about 44 GB of output):

```bash
python -m pip install h5py
python third_party/dust3r/datasets_preprocess/preprocess_megadepth.py \
  --megadepth_dir /path/to/megadepth/phoenix/S6/zl548/MegaDepth_v1 \
  --precomputed_pairs /path/to/megadepth_pairs.npz \
  --output_dir /path/to/processed_data/megadepth_processed_for_dust3r
cp /path/to/megadepth_pairs.npz /path/to/processed_data/megadepth_processed_for_dust3r/all_metadata.npz
```

### CO3D v2

Download the full CO3D v2 release from [facebookresearch/co3d](https://github.com/facebookresearch/co3d) (51 categories, about 5.5 TB, no registration):

```bash
git clone https://github.com/facebookresearch/co3d /path/to/co3d_code
mkdir -p /path/to/co3d_v2
python /path/to/co3d_code/co3d/download_dataset.py --download_folder /path/to/co3d_v2 \
  --clear_archives_after_unpacking
```

Convert (about 127 GB of output):

```bash
python third_party/dust3r/datasets_preprocess/preprocess_co3d.py \
  --co3d_dir /path/to/co3d_v2 \
  --output_dir /path/to/processed_data/co3d_processed_for_dust3r
```

Leave out `--single_sequence_subset` in both steps. That subset lacks the `fewview_train` set lists the converter reads and the categories microwave, stopsign and tv.

### StaticThings3D

Download `StaticThings3D_3DV22` from LMB Freiburg (about 147 GB, no registration) and Naver's pair file:

```bash
for f in depths frames_cleanpass frames_finalpass intrinsics poses; do
  wget -c -P /path/to/staticthings3d https://lmb.informatik.uni-freiburg.de/data/StaticThings3D_3DV22/$f.tar.bz2 &&
  tar xf /path/to/staticthings3d/$f.tar.bz2 -C /path/to/staticthings3d || break
done
wget -P /path/to https://download.europe.naverlabs.com/ComputerVision/DUSt3R/staticthings_pairs.npy
sha256sum /path/to/staticthings_pairs.npy  # 91287ea2151c78e3606e285ff92eaded4f11529b2fd3b3cbf8cc8163f1b27782
```

Convert (about 46 GB of output):

```bash
python third_party/dust3r/datasets_preprocess/preprocess_staticthings3d.py \
  --StaticThings3D_dir /path/to/staticthings3d \
  --precomputed_pairs /path/to/staticthings_pairs.npy \
  --output_dir /path/to/processed_data/staticthings3d_processed_for_dust3r
cp /path/to/staticthings_pairs.npy /path/to/processed_data/staticthings3d_processed_for_dust3r/
```

### Waymo

Accept the [license](https://waymo.com/open/terms/) at [waymo.com/open](https://waymo.com/open/), then download the v1 `.tfrecord` files of Perception Dataset v1.4.2, training split, from `gs://waymo_open_dataset_v_1_4_2`. You need only the 189 segments named in the pair file:

```bash
wget -P /path/to https://download.europe.naverlabs.com/ComputerVision/DUSt3R/waymo_pairs.npz
sha256sum /path/to/waymo_pairs.npz  # 6d8038a0c5d19dd150762360f76fa2a385117f19e846d1e3a6b9c7f482ca9395
python -c "import numpy as np; print(*np.load('/path/to/waymo_pairs.npz')['scenes'], sep='\n')"  # the 189 .tfrecord names
```

Convert in a separate environment with Python 3.8-3.10, because the Waymo wheel needs `tensorflow==2.12` and `numpy==1.23`. Install `gcsfs waymo-open-dataset-tf-2-12-0==1.6.4`, plus torch, torchvision, scipy, opencv-python, Pillow and tqdm in versions that work with numpy 1.23. Put the `.tfrecord` files in one directory, then run this (about 24 GB of output):

```bash
python third_party/dust3r/datasets_preprocess/preprocess_waymo.py \
  --waymo_dir /path/to/waymo_tfrecords \
  --precomputed_pairs /path/to/waymo_pairs.npz \
  --output_dir /path/to/processed_data/waymo_processed_for_dust3r \
  --workers 8
cp /path/to/waymo_pairs.npz /path/to/processed_data/waymo_processed_for_dust3r/
```

### ARKitScenes

Download DUSt3R's pair archive (about 290 MB) and the `raw` dataset with Apple's [`download_data.py`](https://github.com/apple/ARKitScenes/blob/main/download_data.py) (3,493 scenes, about 3.6 TB, no registration). The Python step drops six scenes that are not in our index and links `raw/Test` to the matching scenes in `raw/Training`:

```bash
wget -P /path/to https://download.europe.naverlabs.com/ComputerVision/DUSt3R/arkitscenes_pairs.zip
sha256sum /path/to/arkitscenes_pairs.zip  # d140cc1ce124e72354d5f63617180a497915f69135f18bdc2813ed117dedd738
unzip -q -d /path/to /path/to/arkitscenes_pairs.zip  # creates /path/to/arkitscenes_pairs/{Training,Test}
python - /path/to/arkitscenes_pairs /path/to/arkitscenes <<'EOF'
import csv, json, os, sys, numpy as np
pairs, ark = sys.argv[1:3]
drop = {"47332918", "47670174", "47895197", "47895310", "48018780", "48458690"}
lists = {sp: [s for s in json.load(open(f"{pairs}/{sp}/scene_list.json")) if s not in drop] for sp in ("Training", "Test")}
json.dump(lists["Training"], open(f"{pairs}/Training/scene_list.json", "w"))
ok = lambda d: len(d["selection"]) and len(d["pairs"])
ids = dict.fromkeys(s for sp in lists for s in lists[sp] if ok(np.load(f"{pairs}/{sp}/{s}/selected_pairs.npz")))
os.makedirs(f"{ark}/raw/Test", exist_ok=True)
csv.writer(open(f"{ark}/download.csv", "w")).writerows([("video_id", "fold")] + [(s, "Training") for s in ids])
for s in lists["Test"]: os.symlink(f"../Training/{s}", f"{ark}/raw/Test/{s}")
print(len(ids))  # 3493
EOF
git clone https://github.com/apple/ARKitScenes /path/to/ARKitScenes_code && pip install pandas
python /path/to/ARKitScenes_code/download_data.py raw --video_id_csv /path/to/arkitscenes/download.csv \
  --download_dir /path/to/arkitscenes \
  --raw_dataset_assets vga_wide vga_wide_intrinsics lowres_depth lowres_wide.traj 2>&1 | tee download.log
grep -c Error download.log  # must print 0
```

Convert (about 168 GB of output):

```bash
pip install numpy-quaternion==2024.0.13
python third_party/dust3r/datasets_preprocess/preprocess_arkitscenes.py \
  --arkitscenes_dir /path/to/arkitscenes/raw --precomputed_pairs /path/to/arkitscenes_pairs \
  --output_dir /path/to/processed_data/arkitscenes_processed_for_dust3r
```

`scripts/audit_datasets.py` will likely report a different `all_metadata.npz` (float noise in `trajectories`). Check `hashlib.sha256(np.load(path)[key].tobytes()).hexdigest()` of `images` and `pairs` instead:

| Folder | Images | Pairs | sha256 of `images` | sha256 of `pairs` |
|---|---:|---:|---|---|
| `Training` | 955,970 | 37,741,093 | `ce122fa86f233e0b454d4667fc4c591957d4adb92830d9d19242f1384c4618fa` | `6d953acf42050c8f80dec314eff8b3d2cd53da2ffaf895b5bcc855f6d01243ad` |
| `Test` | 48,774 | 1,969,545 | `79f38f85c82a8ab25f059fd1b44a943ef6423e505bfffa9616a0a6efc690b89d` | `cd2570f2f4283c8b58b6c38b6c052a858d5db1c5392af26e2ce56c316e6d1692` |

### WildRGB-D

Download [`hongchi/wildrgbd`](https://huggingface.co/hongchi/wildrgbd) with the `download.py` of [`wildrgbd/wildrgbd`](https://github.com/wildrgbd/wildrgbd) (about 3.4 TB of archives, 4 TB extracted, no registration):

```bash
git clone https://github.com/wildrgbd/wildrgbd /path/to/wildrgbd_code
mkdir -p /abs/path/wildrgbd
(cd /abs/path/wildrgbd && python /path/to/wildrgbd_code/download.py --cat all)
ls -d /abs/path/wildrgbd/*/scenes | wc -l  # expect 46
```

`download.py` does not stop on errors, and missing scenes change the selected sequences, so check that all 46 categories extracted fully.

Convert (about 38 GB of output). Use absolute paths: a relative `--wildrgbd_dir` doubles the path prefix, skips every sequence and writes empty split files.

```bash
python third_party/dust3r/datasets_preprocess/preprocess_wildrgbd.py \
  --wildrgbd_dir /abs/path/wildrgbd \
  --output_dir /abs/path/processed_data/wildrgbd_processed_for_dust3r
```

After a failure, delete the `selected_seqs_*.json` files before rerunning, because the script skips any split or category that has one.

### Habitat

Request access to HM3D on Matterport's [HM3D page](https://matterport.com/partners/meta) (free account: Settings > Developer Tools > Request Access under "Habitat - Matterport 3D Research Dataset"), then create an API token under [Developer Tools](https://my.matterport.com/settings/account/devtools).

For ScanNet, email the signed [ScanNet Terms of Use](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf) from an institutional address to scannet@googlegroups.com, get ScanNet v2 from [scan-net.org](http://www.scan-net.org/), then run `assimp export <PLY FILE> <GLB PATH>` per scene to get `scannet/<sceneXXXX_YY>.glb` under `/path/to/habitat/data/scene_datasets`.

The commands below fetch the rest (about 33 GB). You do not need Gibson.

```bash
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/habitat_5views_v1_512x512_metadata.tar.gz
tar -xzf habitat_5views_v1_512x512_metadata.tar.gz -C /path/to/habitat  # -> 5views_v1_512x512_metadata/
for s in train val minival; do
  curl --location --user <api-token-id>:<api-token-secret> -o hm3d-$s-habitat-v0.2.tar \
    https://api.matterport.com/resources/habitat/hm3d-$s-habitat-v0.2.tar
  mkdir -p /path/to/habitat/data/scene_datasets/hm3d/$s
  tar -xf hm3d-$s-habitat-v0.2.tar -C /path/to/habitat/data/scene_datasets/hm3d/$s
done
git lfs install
git clone https://huggingface.co/datasets/ai-habitat/ReplicaCAD_baked_lighting.git /path/to/habitat/data/scene_datasets/ReplicaCAD_baked_lighting
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /path/to/habitat/data/  # needs the habitat env below
```

Convert (needs a GPU with EGL and GNU `parallel`, see the [bundled README](../third_party/dust3r/datasets_preprocess/habitat/README.md)):

```bash
conda create -n habitat python=3.8 habitat-sim=0.2.1 headless=2.0 -c aihabitat -c conda-forge
conda activate habitat && conda install pytorch -c pytorch && pip install opencv-python tqdm
export METADATA_DIR=/path/to/habitat/5views_v1_512x512_metadata SCENES_DIR=/path/to/habitat/data/scene_datasets
export OUTPUT_DIR=/path/to/processed_data/habitat_processed_for_dust3r
cd third_party/dust3r/datasets_preprocess/habitat && export PYTHONPATH=$PWD
python preprocess_habitat.py --scenes_dir=$SCENES_DIR --metadata_dir=$METADATA_DIR --output_dir=$OUTPUT_DIR \
  | grep -v '/gibson/' | parallel -j 16
cd -  # back to the repository root
cp click3r/configs/habitat/Habitat_100000_scenes_train.txt $OUTPUT_DIR/
cp click3r/configs/habitat/Habitat_10000_scenes_val.txt $OUTPUT_DIR/
```

Delete partially rendered scenes before rerunning, because `preprocess_habitat.py` skips scenes whose output directory exists. If you hit a headless or EGL error, use the README's fallback, which builds habitat-sim from the `stable` tag.

### ScanNet++

[Register](https://scannetpp.mlsg.cit.tum.de/scannetpp/register) with an institutional email, log in, create an application and submit the signed terms as shown after login. After approval, use the token and download script on your dashboard. The default download (about 1.5 TB) has every file needed, and only the 229 scenes in `scene_list.json` of the DUSt3R v1 pair list are used:

```bash
wget -P /path/to https://download.europe.naverlabs.com/ComputerVision/DUSt3R/scannetpp_pairs.zip
sha256sum /path/to/scannetpp_pairs.zip  # 79d68375808bca6059223d14a8150ba3fda78ce8348bb103d000437d71af5e42
unzip /path/to/scannetpp_pairs.zip -d /path/to  # scannetpp_pairs/{scene_list.json,<scene>/selected_pairs.npz}
```

To convert, first decode the iPhone videos with the [ScanNet++ toolkit](https://github.com/scannetpp/scannetpp) (commit `770d03d` or later, `ffmpeg` on PATH). Run its `pip install -r requirements.txt` in a separate environment, because it pins `torch==1.13.1+cu116`. The output is about 14 GB.

```bash
# in the toolkit checkout; /path/to/scannetpp holds data/
cp iphone/configs/prepare_iphone_data.yml.template iphone/configs/prepare_iphone_data.yaml
python -c "import json; print(*json.load(open('/path/to/scannetpp_pairs/scene_list.json')), sep='\n')" > dust3r_scenes.txt
python -m iphone.prepare_iphone_data data_root=/path/to/scannetpp extract_depth=false \
  +scene_list_file=$PWD/dust3r_scenes.txt +extract_only_rgb_in_colmap=true +output_root=null

# in the Click3R root
python -m pip install pyrender
PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/dust3r python third_party/dust3r/datasets_preprocess/preprocess_scannetpp.py \
  --scannetpp_dir /path/to/scannetpp --precomputed_pairs /path/to/scannetpp_pairs \
  --output_dir /path/to/processed_data/scannetpp_processed_for_dust3r --pyopengl-platform egl
```

The converter drops frames with no COLMAP pose (1,189 in our raw data), so your `all_metadata.npz` may differ from ours.

If your raw data has `rgb.mp4` (v1), use a toolkit commit before `e41bbdd`, such as `9191473`. It takes a config file instead of overrides: copy the template to `.yml`, set `data_root`, `scene_list_file` and `extract_depth: false`, and run `python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml`.

## Held-out Table 1 data

Table 1 evaluates on the official test and validation splits of three datasets.

| Dataset | Split | Output |
| --- | --- | --- |
| [CO3D v2](#co3d-v2-test-sequences) | 499 test sequences, 100,013 frames | 24 GB |
| [ARKitScenes](#arkitscenes-validation-fold) | 154 Validation videos, 48,648 frames, 2,142,494 pairs | 8 GB |
| [ScanNet++](#scannet-nvs_sem_val) | 50 `nvs_sem_val` scenes, 13,930 frames, 741,280 pairs | 3 GB |

Run each command from the repository root with the same `--out`. It converts the data with DUSt3R's converter, checks it against [`data_checksums.json`](../click3r/configs/table1/data_checksums.json) and prints `<dataset>: OK, ...`. You can rerun an interrupted command.

### CO3D v2 test sequences

The command reads only these sequences from the public CO3D v2 zips (about 94 GB, no registration). With the full download from [CO3D v2](#co3d-v2), add `--raw /path/to/co3d_v2` to read them from disk instead.

```bash
python scripts/table1/prepare_data.py --dataset co3d --out /path/to/table1_data
```

### ARKitScenes Validation fold

The command reads only the listed frames from Apple's `raw` dataset (about 21 GB, no registration).

```bash
python -m pip install numpy-quaternion==2024.0.13
python scripts/table1/prepare_data.py --dataset arkitscenes --out /path/to/table1_data
```

### ScanNet++ nvs_sem_val

Download ScanNet++ v1, the 50 scenes of `splits/nvs_sem_val.txt`, with the download script from your dashboard (access as in [ScanNet++](#scannet)). Per scene the command reads `dslr/{colmap,resized_images,resized_anon_masks}`, `iphone/{colmap,rgb.mp4,rgb_mask.mkv}` and `scans/mesh_aligned_0.05.ply`. It needs `ffmpeg` on PATH and a GPU with EGL. `--raw` is the folder that holds `data/<scene>/`.

```bash
python -m pip install pyrender
python scripts/table1/prepare_data.py --dataset scannetpp \
  --raw /path/to/scannetpp --out /path/to/table1_data
```

## Hard3DBench

The sample lists are in [`click3r/configs/hard3dbench/`](../click3r/configs/hard3dbench/):

| File | Samples | Each sample lists |
|---|---:|---|
| `two_view_indist.json` | 404 pairs | dataset, `pair_idx`, scene, two frames, RGB paths |
| `two_view_ood.json` | 300 pairs | dataset, scene, two frames, RGB and depth paths |
| `five_view_indist.json` | 400 tuples | dataset, scene, five frames |
| `five_view_ood.json` | 300 tuples | dataset, scene, five frames |

### In-Distribution samples

These come from BlendedMVS (`val`), Habitat (`val`) and WildRGB-D (`test`), converted as above, and from the held-out [CO3D test sequences](#co3d-v2-test-sequences), not the training CO3D conversion. The evaluators read the first three from `--data-root` and CO3D from `--co3d-root` ([Table 2 commands](eval.md#table-2-hard3dbench)).

If you load the lists yourself:

- Use the listed frames, not `dataset[pair_idx]`. The CO3D and WildRGB-D loaders replace a frame whose masked depth is empty with a nearby frame, so 5 CO3D pairs use frames other than the ones `pair_idx` points to.
- RGB paths are relative to the dataset's `<name>_processed_for_dust3r` folder.
- 5-view Habitat frames are `<viewpoint>_<view>` with a 0-based view, so frame `485_3` is the file `485_4.jpeg`.

### Out-of-Distribution datasets

Paths in `two_view_ood.json` are relative to one data root, `/path/to/hard3dbench_ood`.

| Dataset | 2-view pairs | 5-view tuples | Terms | Access |
|---|---:|---:|---|---|
| [7Scenes](#7scenes) | 47 | 60 | [Microsoft Research License (non-commercial)](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) | accept license |
| [DTU](#dtu) | 106 | 60 | [free citeware](https://roboimagedata.compute.dtu.dk/) | open |
| [ETH3D](#eth3d) | 81 | 60 | [CC BY-NC-SA 4.0](https://www.eth3d.net/) | open |
| [NRGBD](#nrgbd) | 34 | 60 | [per-scene CC BY / CC0](https://github.com/dazinovic/neural-rgbd-surface-reconstruction) | open |
| [ScanNetv2](#scannetv2) | 32 | 60 | [ScanNet Terms of Use](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf) | signed agreement |

Run each command from the repository root. It writes the listed frames and their cameras, checks them against [`ood_checksums.json`](../click3r/configs/hard3dbench/ood_checksums.json) and prints `<dataset>: OK, ...`.

#### 7Scenes

Download the seven scene zips from the [project page](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) (about 22 GB, no registration):

```bash
B=https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8
for s in chess fire heads office pumpkin redkitchen stairs; do wget -P /path/to/7scenes_zips $B/$s.zip; done
```

Then pass the folder with the zips as `--raw`, without extracting them:

```bash
python scripts/hard3dbench/prepare_ood.py --dataset 7scenes \
  --raw /path/to/7scenes_zips --out /path/to/hard3dbench_ood
```

#### DTU

Download only `dtu_test_mvsnet.zip` (about 1 GB, no registration) from Spann3R's [Google Drive folder](https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy). Only this zip has the depth maps Spann3R rendered, which the lists use.

```bash
md5sum /path/to/dtu_test_mvsnet.zip  # 5688301ad96a7118d5f63da22e69b61b
```

Then pass the zip itself as `--raw`:

```bash
python scripts/hard3dbench/prepare_ood.py --dataset dtu \
  --raw /path/to/dtu_test_mvsnet.zip --out /path/to/hard3dbench_ood
```

#### ETH3D

Download the distorted JPG images and the per-scene depth archives from the [ETH3D datasets page](https://www.eth3d.net/datasets) (about 11 GB, no registration). Extracting needs `7z`.

```bash
D=/path/to/eth3d; mkdir -p $D
wget -P $D https://www.eth3d.net/data/multi_view_training_dslr_jpg.7z
for s in courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains; do
  wget -P $D https://www.eth3d.net/data/${s}_dslr_depth.7z
done
for f in $D/*.7z; do 7z x "$f" -o"$D" && rm "$f"; done
```

Then pass the extracted folder as `--raw`. The output is about 24 GB.

```bash
python scripts/hard3dbench/prepare_ood.py --dataset eth3d \
  --raw /path/to/eth3d --out /path/to/hard3dbench_ood
```

#### NRGBD

Download `neural_rgbd_data.zip` from the [Neural RGB-D README](https://github.com/dazinovic/neural-rgbd-surface-reconstruction#dataset) (about 7.8 GB, no registration):

```bash
wget -P /path/to https://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip
```

Then pass the zip itself as `--raw`:

```bash
python scripts/hard3dbench/prepare_ood.py --dataset nrgbd \
  --raw /path/to/neural_rgbd_data.zip --out /path/to/hard3dbench_ood
```

#### ScanNetv2

Request access through the Terms of Use process in the [ScanNet README](https://github.com/ScanNet/ScanNet#scannet-data): send the filled-out agreement from your institutional email to scannet@googlegroups.com. After approval you receive `download-scannet.py`.

Download the `.sens` files of these 33 test scenes (about 34 GB). The script asks you to confirm the Terms of Use on every run, so expect one prompt per scene.

```bash
for n in 0708 0710 0720 0721 0723 0725 0726 0729 0730 0731 0740 0742 0750 0754 0755 0760 0763 \
         0764 0768 0769 0775 0776 0777 0780 0786 0788 0792 0795 0796 0797 0798 0802 0805; do
  python download-scannet.py -o /path/to/scannet --id scene${n}_00 --type .sens
done
```

Then pass the same folder as `--raw`:

```bash
python scripts/hard3dbench/prepare_ood.py --dataset scannetv2 \
  --raw /path/to/scannet --out /path/to/hard3dbench_ood
```
