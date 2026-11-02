"""Derived Toy2Box observations for BC-RNN training and playback."""

import h5py
import numpy as np
from torch.utils.data import default_collate


#########################
# relative position
#########################

def eef_to_toy(obs):
    """Return the XYZ vector from the gripper tip to the toy in the robot base frame, in metres."""
    return obs["toy_truck_positions"] - obs["eef_pos"]


def collate_with_eef_to_toy(samples):
    batch = default_collate(samples)
    for key in ("obs", "next_obs", "goal_obs"):
        if key in batch:
            batch[key]["eef_to_toy"] = eef_to_toy(batch[key])
    return batch


#########################
# normalization statistics
#########################

def eef_to_toy_normalization_stats(dataset):
    """Compute statistics over unpadded frames in the selected training demos."""
    count = 0
    mean = np.zeros((1, 3), dtype=np.float64)
    sqdiff = np.zeros_like(mean)
    with h5py.File(dataset.hdf5_path, "r", swmr=dataset.hdf5_use_swmr) as handle:
        for demo_name in dataset.demos:
            obs = handle[f"data/{demo_name}/obs"]
            values = eef_to_toy({key: obs[key][()].astype(np.float32)
                                 for key in ("toy_truck_positions", "eef_pos")}).astype(np.float64)
            demo_count = len(values)
            demo_mean = values.mean(axis=0, keepdims=True)
            delta = demo_mean - mean
            total = count + demo_count
            sqdiff += ((values - demo_mean) ** 2).sum(axis=0, keepdims=True)
            sqdiff += delta ** 2 * count * demo_count / total
            mean += delta * demo_count / total
            count = total
    return {
        "mean": mean.astype(np.float32),
        "std": (np.sqrt(sqdiff / count) + 1e-3).astype(np.float32),
    }
