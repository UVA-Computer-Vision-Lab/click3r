"""Hash the compact training indices and split files of the processed datasets."""

import argparse
import hashlib
import json
from pathlib import Path

INDICES = {
    "blendedmvs": ["blendedmvs_pairs.npy"],
    "megadepth": ["all_metadata.npz"],
    "co3d": ["selected_seqs_train.json", "selected_seqs_test.json"],
    "staticthings3d": ["staticthings_pairs.npy"],
    "waymo": ["waymo_pairs.npz"],
    "arkitscenes": ["Training/all_metadata.npz", "Test/all_metadata.npz"],
    "wildrgbd": ["selected_seqs_train.json", "selected_seqs_test.json"],
    "habitat": ["Habitat_100000_scenes_train.txt", "Habitat_10000_scenes_val.txt"],
    "scannetpp": ["all_metadata.npz"],
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root):
    records = {}
    for dataset, names in INDICES.items():
        directory = root / f"{dataset}_processed_for_dust3r"
        records[dataset] = {
            "directory_present": directory.is_dir(),
            "broken_symlink": directory.is_symlink() and not directory.exists(),
            "indices": [
                {
                    "file": name,
                    "present": (directory / name).is_file(),
                    "sha256": sha256(directory / name)
                    if (directory / name).is_file()
                    else None,
                }
                for name in names
            ],
        }
    return {
        "datasets": records,
        "scope": "indices and split files; does not verify all image/depth files",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
