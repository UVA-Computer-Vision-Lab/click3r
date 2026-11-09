"""Convert DUSt3R image-loader records to single-pair model inputs."""

import numpy as np
import torch


def prepare_model_view(view, device, *, keep_numpy_shape_batch=False):
    """Add a batch dimension to geometry fields and preserve image metadata.

    ``load_images`` already batches ``img`` and ``true_shape``. By default a
    NumPy ``true_shape`` gets another dimension, the layout used for Click3R and
    the baselines; ``keep_numpy_shape_batch`` keeps its existing batch dimension,
    the layout used for the original DUSt3R model. ``idx`` must be a list for the
    global aligner.
    """
    result = {}
    for key, value in view.items():
        if torch.is_tensor(value):
            result[key] = (value if key == "img" else value.unsqueeze(0)).to(device)
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
            if not (key == "true_shape" and keep_numpy_shape_batch):
                tensor = tensor.unsqueeze(0)
            result[key] = tensor.to(device)
        else:
            result[key] = [value] if key == "idx" else value
    return result
