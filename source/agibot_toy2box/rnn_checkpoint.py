"""BC-RNN training checkpoints compatible with robomimic playback."""

import os
import random
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from robomimic.utils import file_utils as FileUtils, tensor_utils as TensorUtils


#########################
# random state
#########################

def capture_rng_state(device):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
    }


def restore_rng_state(state, device):
    random.setstate(state["python"])
    algorithm, values, position, has_gauss, cached = state["numpy"]
    np.random.set_state((algorithm, np.asarray(values, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if torch.device(device).type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


#########################
# experiment directories
#########################

def get_output_dirs(output_dir, checkpoint_path=None):
    """Return output, log and model directories; explicit destinations use a flat layout."""
    if output_dir is None:
        checkpoint_dir = Path(checkpoint_path).expanduser().resolve().parent
        if checkpoint_dir.name == "models" and (checkpoint_dir.parent / "logs").is_dir():
            run_dir = checkpoint_dir.parent
            return str(run_dir), str(run_dir / "logs"), str(checkpoint_dir)
        output_dir = checkpoint_dir
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir), str(output_dir), str(output_dir)


#########################
# checkpoint loading
#########################

def load_resume_checkpoint(path):
    path = Path(path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model", "config", "algo_name", "env_metadata", "shape_metadata"}
    if not required.issubset(checkpoint):
        raise ValueError("--resume requires a robomimic checkpoint, not an ACT checkpoint.")
    if "epoch" not in checkpoint:
        match = re.search(r"(?:^|_)epoch_(\d+)(?:_|\.|$)", path.name)
        if match is None:
            raise ValueError("Legacy checkpoint filenames must contain epoch_<number>, e.g. model_epoch_1000.pth.")
        checkpoint["epoch"] = int(match.group(1))
    if not isinstance(checkpoint["epoch"], int) or checkpoint["epoch"] < 0:
        raise ValueError("Checkpoint epoch must be a nonnegative integer.")
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    if config.train.hdf5_normalize_obs and "obs_normalization_stats" not in checkpoint:
        raise ValueError("The checkpoint is missing its required observation normalization statistics.")
    return config, checkpoint


def restore_training_state(model, checkpoint):
    model.deserialize(checkpoint["model"])
    if "optimizers" not in checkpoint:
        print("[RESUME] Legacy checkpoint: model weights restored; optimizer and LR scheduler start fresh.")
        return
    if set(checkpoint["optimizers"]) != set(model.optimizers):
        raise ValueError("Checkpoint optimizer names do not match the model.")
    if set(checkpoint["lr_schedulers"]) != set(model.lr_schedulers):
        raise ValueError("Checkpoint LR scheduler names do not match the model.")
    for key, optimizer in model.optimizers.items():
        optimizer.load_state_dict(checkpoint["optimizers"][key])
    for key, scheduler in model.lr_schedulers.items():
        state = checkpoint["lr_schedulers"][key]
        if (scheduler is None) != (state is None):
            raise ValueError(f"Checkpoint LR scheduler {key} does not match the model.")
        if scheduler is not None:
            scheduler.load_state_dict(state)
    restore_rng_state(checkpoint["rng_state"], model.device)
    print("[RESUME] Model, optimizer, LR scheduler and random states restored.")


#########################
# checkpoint saving
#########################

def save_training_checkpoint(model, config, env_meta, shape_meta, ckpt_path, epoch,
                             best_valid_loss, obs_normalization_stats=None):
    checkpoint = {
        "model": model.serialize(),
        "config": config.dump(),
        "algo_name": config.algo_name,
        "env_metadata": deepcopy(env_meta),
        "shape_metadata": deepcopy(shape_meta),
        "epoch": epoch,
        "optimizers": {key: optimizer.state_dict() for key, optimizer in model.optimizers.items()},
        "lr_schedulers": {key: scheduler.state_dict() if scheduler is not None else None
                          for key, scheduler in model.lr_schedulers.items()},
        "rng_state": capture_rng_state(model.device),
        "best_valid_loss": best_valid_loss,
    }
    if obs_normalization_stats is not None:
        checkpoint["obs_normalization_stats"] = TensorUtils.to_list(deepcopy(obs_normalization_stats))
    temporary_path = str(ckpt_path) + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, ckpt_path)
    print(f"save checkpoint to {ckpt_path}")
