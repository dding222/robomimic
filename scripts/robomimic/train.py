# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MIT License
#
# Copyright (c) 2021 Stanford Vision and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
The main entry point for training policies from pre-collected data.

This script loads dataset(s), creates a model based on the algorithm specified,
and trains the model. It supports training on various environments with multiple
algorithms from robomimic.

Args:
    algo: Name of the algorithm to run.
    task: Name of the environment.
    name: If provided, override the experiment name defined in the config.
    dataset: If provided, override the dataset path defined in the config.
    seed: If provided, override the JSON seed for a new training run.
    output_dir: Directory to save models, configuration and logs directly.
    normalize_training_actions: Whether to normalize actions in the training data.

This file has been modified from the original robomimic version to integrate with IsaacLab.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse
import importlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import psutil
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
import torch
from robomimic.algo import algo_factory
from robomimic.config import Config, config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger
from torch.utils.data import DataLoader

import isaaclab_tasks  # noqa: F401
import agibot_toy2box  # noqa: F401
from agibot_toy2box.rnn_checkpoint import (
    get_output_dirs, load_resume_checkpoint, restore_training_state, save_training_checkpoint,
)
from agibot_toy2box.rnn_observations import collate_with_eef_to_toy, eef_to_toy_normalization_stats
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


def normalize_hdf5_actions(config: Config, log_dir: str) -> str:
    """Normalizes actions in hdf5 dataset to [-1, 1] range.

    Args:
        config: The configuration object containing dataset path.
        log_dir: Directory to save normalization parameters.

    Returns:
        Path to the normalized dataset.
    """
    base, ext = os.path.splitext(config.train.data)
    normalized_path = base + "_normalized" + ext

    # Copy the original dataset
    print(f"Creating normalized dataset at {normalized_path}")
    shutil.copyfile(config.train.data, normalized_path)

    # Open the new dataset and normalize the actions
    with h5py.File(normalized_path, "r+") as f:
        dataset_paths = [f"/data/demo_{str(i)}/actions" for i in range(len(f["data"].keys()))]

        # Compute the min and max of the dataset
        dataset = np.array(f[dataset_paths[0]]).flatten()
        for i, path in enumerate(dataset_paths):
            if i != 0:
                data = np.array(f[path]).flatten()
                dataset = np.append(dataset, data)

        max = np.max(dataset)
        min = np.min(dataset)

        # Normalize the actions
        for i, path in enumerate(dataset_paths):
            data = np.array(f[path])
            normalized_data = 2 * ((data - min) / (max - min)) - 1  # Scale to [-1, 1] range
            del f[path]
            f[path] = normalized_data

        # Save the min and max values to log directory
        with open(os.path.join(log_dir, "normalization_params.txt"), "w") as f:
            f.write(f"min: {min}\n")
            f.write(f"max: {max}\n")

    return normalized_path


def train(config: Config, device: str, log_dir: str, ckpt_dir: str, resume_checkpoint=None):
    """Train a model using the algorithm specified in config.

    Args:
        config: Configuration object.
        device: PyTorch device to use for training.
        log_dir: Directory to save logs.
        ckpt_dir: Directory to save checkpoints.
    """
    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)
    start_epoch = resume_checkpoint["epoch"] if resume_checkpoint is not None else 0

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

    print(f">>> Saving logs into directory: {log_dir}")
    print(f">>> Saving checkpoints into directory: {ckpt_dir}")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    print(f"Training seed={config.train.seed}")

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    use_eef_to_toy = "eef_to_toy" in config.all_obs_keys
    dataset_obs_keys = [key for key in config.all_obs_keys if key != "eef_to_toy"]
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data, all_obs_keys=dataset_obs_keys, verbose=True
    )
    if use_eef_to_toy:
        for key in ("toy_truck_positions", "eef_pos"):
            if shape_meta["all_shapes"].get(key) != [3]:
                raise ValueError(f"eef_to_toy requires the 3D observation {key}.")
        shape_meta["all_shapes"]["eef_to_toy"] = [3]
        shape_meta["all_shapes"] = OrderedDict(sorted(shape_meta["all_shapes"].items()))
        shape_meta["all_obs_keys"] = list(config.all_obs_keys)
        print("Derived obs: eef_to_toy = toy_truck_positions - eef_pos, shape=(3,), units=m")

    if resume_checkpoint is not None:
        saved_shapes = resume_checkpoint["shape_metadata"]
        expected = {key: tuple(shape) for key, shape in saved_shapes["all_shapes"].items()}
        actual = {key: tuple(shape) for key, shape in shape_meta["all_shapes"].items()}
        if actual != expected or shape_meta["ac_dim"] != saved_shapes["ac_dim"]:
            raise ValueError("Dataset observation/action dimensions do not match the resumed checkpoint.")
        shape_meta = saved_shapes
        print(f"[RESUME] Saved epoch={start_epoch}; training epochs {start_epoch + 1}..{config.train.num_epochs}.")

    state_keys = list(config.observation.modalities.obs.low_dim)
    state_dim = sum(int(np.prod(shape_meta["all_shapes"][key])) for key in state_keys)
    print(f"BC-RNN state_dim={state_dim}, normalize_obs={config.train.hdf5_normalize_obs}")
    print(f"state_keys={state_keys}")

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env
            print(envs[env.name])

    print("")

    # setup for a new training run
    data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # save the config as a json file
    with open(os.path.join(config.train.output_dir, "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=dataset_obs_keys)
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retrieve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        if resume_checkpoint is not None:
            obs_normalization_stats = {
                key: {field: np.asarray(values, dtype=np.float32) for field, values in stats.items()}
                for key, stats in resume_checkpoint["obs_normalization_stats"].items()
            }
        else:
            obs_normalization_stats = trainset.get_obs_normalization_stats()
            if use_eef_to_toy:
                obs_normalization_stats["eef_to_toy"] = eef_to_toy_normalization_stats(trainset)

    # initialize data loaders
    collate_fn = collate_with_eef_to_toy if use_eef_to_toy else None
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )
    else:
        valid_loader = None

    # main training loop
    best_valid_loss = resume_checkpoint.get("best_valid_loss") if resume_checkpoint is not None else None
    last_ckpt_time = time.time()
    if resume_checkpoint is not None:
        restore_training_state(model, resume_checkpoint)
        del resume_checkpoint

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(
            model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = f"model_epoch_{epoch}"

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            last_epoch_check = epoch == config.train.num_epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check or last_epoch_check
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print(f"Train Epoch {epoch}")
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps,
                    obs_normalization_stats=obs_normalization_stats,
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            print(f"Validation Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            save_training_checkpoint(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                epoch=epoch,
                best_valid_loss=best_valid_loss,
                obs_normalization_stats=obs_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

    # terminate logging
    data_logger.close()


#########################
# command line
#########################

def parse_bool(value):
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError("Use True or False.")


def main(args: argparse.Namespace):
    """Train a model on a task using a specified algorithm.

    Args:
        args: Command line arguments.
    """
    # load config
    resume_checkpoint = None
    if args.resume is not None:
        config, resume_checkpoint = load_resume_checkpoint(args.resume)
        if args.algo is not None and args.algo != config.algo_name:
            raise ValueError("--algo must match the resumed checkpoint algorithm.")
        if args.normalize_training_actions:
            raise ValueError("Resume uses the saved action scaling. Use its dataset without --normalize_training_actions.")
        print(f"[RESUME] Loading {Path(args.resume).expanduser().resolve()}")
    elif args.task is not None:
        # obtain the configuration entry point
        cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"
        task_name = args.task.split(":")[-1]

        print(f"Loading configuration for task: {task_name}")
        print(gym.envs.registry.keys())
        print(" ")
        cfg_entry_point_file = gym.spec(task_name).kwargs.pop(cfg_entry_point_key)
        # check if entry point exists
        if cfg_entry_point_file is None:
            raise ValueError(
                f"Could not find configuration for the environment: '{task_name}'."
                f" Please check that the gym registry has the entry point: '{cfg_entry_point_key}'."
            )

        # resolve module path if needed
        if ":" in cfg_entry_point_file:
            mod_name, file_name = cfg_entry_point_file.split(":")
            mod = importlib.import_module(mod_name)
            if mod.__file__ is None:
                raise ValueError(f"Could not find module file for: '{mod_name}'")
            mod_path = os.path.dirname(mod.__file__)
            config_file = os.path.join(mod_path, file_name)
        else:
            config_file = cfg_entry_point_file

        with open(config_file) as f:
            ext_cfg = json.load(f)
            config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        raise ValueError("Please provide a task name through CLI arguments.")

    with config.values_unlocked():
        if args.dataset is not None:
            config.train.data = args.dataset
        if args.name is not None:
            config.experiment.name = args.name
        if args.epochs is not None:
            config.train.num_epochs = args.epochs
        if args.seed is not None:
            if not 0 <= args.seed < 2**32:
                raise ValueError("--seed must be between 0 and 2**32 - 1.")
            if resume_checkpoint is not None and args.seed != config.train.seed:
                raise ValueError("--seed must match the resumed checkpoint. Start a new run to change the seed.")
            config.train.seed = args.seed
        if args.hdf5_normalize_obs is not None:
            if resume_checkpoint is not None and args.hdf5_normalize_obs != config.train.hdf5_normalize_obs:
                raise ValueError("Observation normalization must match the resumed checkpoint. Start a new run to change it.")
            config.train.hdf5_normalize_obs = args.hdf5_normalize_obs
        start_epoch = resume_checkpoint["epoch"] if resume_checkpoint is not None else 0
        if config.train.num_epochs <= start_epoch:
            raise ValueError(f"--epochs is the final epoch and must exceed saved epoch {start_epoch}.")

        output_dir = args.output_dir
        if output_dir is None and resume_checkpoint is None:
            output_dir = os.path.join("./logs", "robomimic", args.task)
        output_dir, log_dir, ckpt_dir = get_output_dirs(output_dir, args.resume)
        config.train.output_dir = output_dir

    if args.normalize_training_actions:
        config.train.data = normalize_hdf5_actions(config, log_dir)

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device, log_dir, ckpt_dir, resume_checkpoint=resume_checkpoint)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--algo", type=str, default=None, help="Name of the algorithm.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Training seed. Defaults to the JSON value (101), or the checkpoint seed on resume.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume a robomimic checkpoint using its saved configuration.")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Save models, config.json, log.txt and tb/ directly here. Resume defaults to the checkpoint's existing directory.",
    )
    parser.add_argument("--normalize_training_actions", action="store_true", default=False, help="Normalize actions")
    parser.add_argument(
        "--norm", dest="hdf5_normalize_obs", type=parse_bool, default=None,
        help="Normalize observations with training mean/std. Default: False in the JSON, or the checkpoint setting on resume.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Final training epoch, not additional epochs. Overrides the JSON or resumed checkpoint target."
        ),
    )

    try:
        args = parser.parse_args()
        main(args)
    finally:
        simulation_app.close()
