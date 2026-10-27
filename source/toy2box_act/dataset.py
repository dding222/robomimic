"""Use robomimic's BC-RNN sequence sampling for ACT action chunks."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils.dataset import SequenceDataset


#########################
# dataset inspection
#########################

CAMERA_KEYS = {"eye": "hand_eye_cam", "top": "top_cam", "front": "front_cam"}


def inspect_dataset(path, enable_cam=False, camera_keys=None):
    config_path = Path(__file__).parents[1] / "agibot_toy2box/agents/robomimic/bc_rnn_low_dim.json"
    config = json.loads(config_path.read_text())
    state_keys = config["observation"]["modalities"]["obs"]["low_dim"]
    with h5py.File(path, "r") as handle:
        demos = list(handle["data"].values())
        if not demos:
            raise ValueError("The HDF5 dataset has no demonstrations.")
        first = demos[0]
        if enable_cam:
            if camera_keys is None:
                camera_keys = sorted(k for k, v in first["obs"].items() if v.ndim == 4 and v.shape[-1] == 3)
            if not 1 <= len(camera_keys) <= 3 or len(set(camera_keys)) != len(camera_keys):
                raise ValueError("Select 1 to 3 unique RGB keys with --camera_keys. Record RGB with --record_camera_data.")
            camera_keys = [CAMERA_KEYS.get(key, key) for key in camera_keys]
            if len(set(camera_keys)) != len(camera_keys) or any(key not in CAMERA_KEYS.values() for key in camera_keys):
                raise ValueError("Select unique cameras from eye, top, front with --cameras.")
        else:
            if camera_keys:
                raise ValueError("--camera_keys requires --enable_cam True.")
            camera_keys = []
        shapes = {k: tuple(first["obs"][k].shape[1:]) for k in state_keys if k in first["obs"]}
        for demo in demos:
            length = demo["actions"].shape[0]
            if length == 0 or demo["actions"].shape != (length, 7):
                raise ValueError(f"{demo.name}: expected nonempty (T, 7) robomimic actions.")
            if demo.attrs.get("num_samples", length) != length:
                raise ValueError(f"{demo.name}: num_samples does not match action length.")
            for key in state_keys:
                if key not in shapes or key not in demo["obs"] or demo["obs"][key].shape != (length, *shapes[key]):
                    raise ValueError(f"{demo.name}: missing or inconsistent state observation {key}.")
            for key in camera_keys:
                if key not in demo["obs"]:
                    raise ValueError(f"{demo.name}: missing camera {key}.")
                image = demo["obs"][key]
                if image.ndim != 4 or image.shape[0] != length or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError(f"{demo.name}/{key}: expected uint8 RGB (T, H, W, 3).")
                if image.shape[1:] != first["obs"][key].shape[1:]:
                    raise ValueError(f"{demo.name}/{key}: image size differs between episodes.")
    state_dim = sum(int(np.prod(shapes[key])) for key in state_keys)
    return state_keys, camera_keys, state_dim


#########################
# sequence preprocessing
#########################

def make_dataset(path, state_keys, camera_keys, seq_length=10):
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": state_keys, "rgb": camera_keys}})
    return SequenceDataset(
        hdf5_path=str(path),
        obs_keys=state_keys + camera_keys,
        dataset_keys=["actions"],
        frame_stack=1,
        seq_length=seq_length,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=True,
        hdf5_cache_mode="low_dim",
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        load_next_obs=False,
    )


def prepare_observations(obs, state_keys, camera_keys, device):
    """Accept current (B, ...) observations, including uint8 HWC camera images."""
    state = torch.cat([torch.as_tensor(obs[k]).flatten(start_dim=1) for k in state_keys], dim=-1)
    batch = {"observation.state": state.to(device=device, dtype=torch.float32)}
    if camera_keys:
        images = ObsUtils.process_obs_dict({k: torch.as_tensor(obs[k]) for k in camera_keys})
        batch["observation.images"] = [images[k].to(device=device, dtype=torch.float32) for k in camera_keys]
    return batch


def prepare_batch(sample, state_keys, camera_keys, device):
    obs = {k: v[:, 0] for k, v in sample["obs"].items() if k != "pad_mask"}
    batch = prepare_observations(obs, state_keys, camera_keys, device)
    batch["action"] = sample["actions"].to(device=device, dtype=torch.float32)
    batch["action_is_pad"] = ~sample["pad_mask"].squeeze(-1).to(device=device, dtype=torch.bool)
    return batch
