"""Use robomimic's BC-RNN sequence sampling for ACT action chunks."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import default_collate
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils.dataset import SequenceDataset


#########################
# dataset inspection
#########################

CAMERA_KEYS = {"eye": "hand_eye_cam", "top": "top_cam", "front": "front_cam"}


def rnn_train_config():
    path = Path(__file__).parents[1] / "agibot_toy2box/agents/robomimic/bc_rnn_low_dim.json"
    return json.loads(path.read_text())["train"]


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

class ACTSequenceDataset(SequenceDataset):
    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0,
                                   seq_length=1, prefix="obs"):
        return super().get_obs_sequence_from_demo(
            demo_id, index_in_demo, keys, num_frames_to_stack=num_frames_to_stack,
            seq_length=1, prefix=prefix,
        )


def make_dataset(path, state_keys, camera_keys, action_chunk=10, hdf5_normalize_obs=None):
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": state_keys, "rgb": camera_keys}})
    if hdf5_normalize_obs is None:
        hdf5_normalize_obs = rnn_train_config()["hdf5_normalize_obs"]
    return ACTSequenceDataset(
        hdf5_path=str(path),
        obs_keys=state_keys + camera_keys,
        dataset_keys=["actions"],
        frame_stack=1,
        seq_length=action_chunk,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=True,
        hdf5_cache_mode="low_dim",
        hdf5_use_swmr=True,
        hdf5_normalize_obs=hdf5_normalize_obs,
        load_next_obs=False,
    )


def normalization_stats(dataset, state_keys, camera_keys, normalize_actions=False):
    """Reuse SequenceDataset statistics and local BC's single global action min/max."""
    result = {}
    if dataset.hdf5_normalize_obs:
        obs_stats = dataset.get_obs_normalization_stats()
        result["state"] = {
            field: torch.cat([torch.as_tensor(obs_stats[key][field]).flatten() for key in state_keys])
            for field in ("mean", "std")
        }
        result["images"] = {
            key: {field: torch.as_tensor(obs_stats[key][field][0]) for field in ("mean", "std")}
            for key in camera_keys
        }
    if normalize_actions:
        action_min, action_max = float("inf"), float("-inf")
        for demo in dataset.demos:
            actions = dataset.get_dataset_for_ep(demo, "actions")[:]
            if not np.isfinite(actions).all():
                raise ValueError(f"Non-finite actions in {demo}.")
            action_min = min(action_min, float(actions.min()))
            action_max = max(action_max, float(actions.max()))
        if action_max <= action_min:
            raise ValueError("Action normalization requires nonconstant data.")
        result["action"] = {"min": action_min, "max": action_max}
    return result


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


def collate_batch(samples, state_keys, camera_keys):
    return prepare_batch(default_collate(samples), state_keys, camera_keys, "cpu")


def batch_to_device(batch, device):
    non_blocking = torch.device(device).type == "cuda"
    return {
        key: ([image.to(device, non_blocking=non_blocking) for image in value]
              if key == "observation.images" else value.to(device, non_blocking=non_blocking))
        for key, value in batch.items()
    }
