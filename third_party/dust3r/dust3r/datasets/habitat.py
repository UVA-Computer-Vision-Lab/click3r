# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed habitat
# dataset at https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md
# See datasets_preprocess/habitat for more details
# --------------------------------------------------------
import os.path as osp
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # noqa
import cv2  # noqa
import numpy as np
from PIL import Image
import json

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset


class Habitat(BaseStereoViewDataset):
    def __init__(self, size, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert self.split is not None
        # loading list of scenes
        with open(osp.join(self.ROOT, f'Habitat_{size}_scenes_{self.split}.txt')) as f:
            self.scenes = f.read().splitlines()
        # Cache available views per scene to avoid repeated filesystem checks
        self._view_cache = {}

    def filter_scene(self, label, instance=None):
        if instance:
            subscene, instance = instance.split('_')
            label += '/' + subscene
            self.instances = [int(instance) - 1]
        valid = np.bool_([scene.startswith(label) for scene in self.scenes])
        assert sum(valid), 'no scene was selected for {label=} {instance=}'
        self.scenes = [scene for i, scene in enumerate(self.scenes) if valid[i]]

    def _get_available_views(self, data_path, key):
        """Detect which view indices are available for this viewpoint by checking file existence."""
        cache_key = osp.join(data_path, key)
        if cache_key in self._view_cache:
            return self._view_cache[cache_key]

        # Check which view files exist (views are 1-indexed in filenames)
        available_views = []
        for view_idx in range(1, 6):  # Check up to 5 views
            impath = osp.join(data_path, f"{key}_{view_idx}.jpeg")
            if osp.exists(impath):
                available_views.append(view_idx - 1)  # Store as 0-indexed

        self._view_cache[cache_key] = available_views
        return available_views

    def _get_views(self, idx, resolution, rng):
        # Retry mechanism for scenes with insufficient views (very rare: ~0.001% of data)
        max_retries = 10
        for attempt in range(max_retries):
            # Use the original index on first attempt, then random indices
            current_idx = idx if attempt == 0 else rng.integers(0, len(self.scenes))
            scene = self.scenes[current_idx]
            data_path, key = osp.split(osp.join(self.ROOT, scene))

            # Get available views for this specific viewpoint
            available_views = self._get_available_views(data_path, key)

            # Skip scenes with less than 2 views and retry with a different scene
            # only scannet/scene0528_00/328 has less than 2 (1) view for the 100000 list
            if len(available_views) < 2:
                if attempt == 0:
                    print(f"Warning: Skipping scene with only {len(available_views)} view(s): {scene}, retrying with another scene...")
                continue

            # Found a valid scene, proceed with sampling
            break
        else:
            # All retries exhausted
            raise RuntimeError(f"Failed to find a scene with >= 2 views after {max_retries} attempts")

        # Sample two views: prefer view 0 if available, otherwise random
        if 0 in available_views:
            # Original behavior: view 0 + random other view
            other_views = [v for v in available_views if v != 0]
            two_random_views = [0, rng.choice(other_views)]
        else:
            # View 0 not available, sample two random views
            two_random_views = list(rng.choice(available_views, size=2, replace=False))

        views = []
        for view_index in two_random_views:
            # Load the view (try this view and fallback to other available views if broken)
            loaded = False
            for attempt_view in [view_index] + [v for v in available_views if v != view_index]:
                image, depthmap, intrinsics, camera_pose = self._load_one_view(data_path, key, attempt_view, resolution, rng)
                if np.isfinite(camera_pose).all():
                    view_index = attempt_view  # Use the successfully loaded view
                    loaded = True
                    break

            assert loaded, f"Failed to load any valid view from {available_views} for {scene}"

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='Habitat',
                label=osp.relpath(data_path, self.ROOT),
                instance=f"{key}_{view_index}",
                img_path=osp.join(data_path, f"{key}_{view_index + 1}.jpeg")))  # CLICK3R: needed to collate mixed-dataset batches and for native-resolution evaluation
        return views

    def _load_one_view(self, data_path, key, view_index, resolution, rng):
        view_index += 1  # file indices starts at 1
        impath = osp.join(data_path, f"{key}_{view_index}.jpeg")
        image = Image.open(impath)

        depthmap_filename = osp.join(data_path, f"{key}_{view_index}_depth.exr")
        depthmap = cv2.imread(depthmap_filename, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

        camera_params_filename = osp.join(data_path, f"{key}_{view_index}_camera_params.json")
        with open(camera_params_filename, 'r') as f:
            camera_params = json.load(f)

        intrinsics = np.float32(camera_params['camera_intrinsics'])
        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, :3] = camera_params['R_cam2world']
        camera_pose[:3, 3] = camera_params['t_cam2world']

        image, depthmap, intrinsics = self._crop_resize_if_necessary(
            image, depthmap, intrinsics, resolution, rng, info=impath)
        return image, depthmap, intrinsics, camera_pose


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = Habitat(1_000_000, split='train', ROOT="data/habitat_processed",
                      resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
