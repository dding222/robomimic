# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play BC-RNN checkpoints following IsaacLab_RS robomimic playback.

Rollout evaluation and video recording follow
IsaacLab_RS/scripts/imitation_learning/robomimic/play.py.
The policy loader and observations use the saved robomimic BC-RNN configuration.
Each seed runs num_rollouts episodes, each with at most horizon steps.
Each rollout saves its own video, capped at video_length steps.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

#########################
# command line
#########################

parser = argparse.ArgumentParser(description="Play and evaluate a robomimic BC-RNN policy in Isaac Lab.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video for each rollout.")
parser.add_argument("--video_length", type=int, default=800,
                    help="Maximum length of each recorded rollout video (in steps).")
parser.add_argument("--video_folder", "--video_dir", dest="video_dir", type=str, default="./videos/robomimic",
                    help="Directory for recorded rollout videos. Default: ./videos/robomimic.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Robomimic BC-RNN checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts per seed.")
seed_args = parser.add_mutually_exclusive_group()
seed_args.add_argument("--seed", type=int, default=101, help="Single evaluation seed. Default: 101.")
seed_args.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="Evaluate each seed and report the mean/std of their success rates.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
evaluation_seeds = args_cli.seeds if args_cli.seeds is not None else [args_cli.seed]
if args_cli.num_rollouts < 1 or args_cli.horizon < 1:
    parser.error("--num_rollouts and --horizon must be positive.")
if args_cli.video_length < 1:
    parser.error("--video_length must be positive.")
if any(seed < 0 or seed >= 2**32 for seed in evaluation_seeds):
    parser.error("Seeds must be between 0 and 2**32 - 1.")
if len(set(evaluation_seeds)) != len(evaluation_seeds):
    parser.error("--seeds must contain unique seeds.")
if args_cli.video:
    args_cli.enable_cameras = True

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version
    # installed by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401
import agibot_toy2box  # noqa: F401
from agibot_toy2box.rnn_observations import eef_to_toy
from isaaclab.sim import create_new_stage
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg


#########################
# checkpoint and observations
#########################

def load_policy(checkpoint_path, device):
    """Use saved BC-RNN shapes and normalization, independently of the training JSON."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("algo_name") != "bc":
        raise ValueError("--checkpoint must be a robomimic BC-RNN checkpoint.")
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    if not config.algo.rnn.enabled:
        raise ValueError("The BC checkpoint must have RNN enabled.")
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=checkpoint, device=device)
    obs_shapes = policy.policy.obs_shapes
    state_dim = sum(int(np.prod(obs_shapes[key])) for key in config.observation.modalities.obs.low_dim)
    print(f"[INFO] BC-RNN state_dim={state_dim}, normalize_obs={config.train.hdf5_normalize_obs}")
    print(f"[INFO] Observation keys: {list(obs_shapes)}")
    return policy


def policy_observations(obs, obs_shapes):
    """Select exactly the checkpoint inputs and derive relative position only when required."""
    inputs = dict(obs)
    if "eef_to_toy" in obs_shapes:
        inputs["eef_to_toy"] = eef_to_toy(inputs)
    result = {}
    for key, shape in obs_shapes.items():
        value = inputs[key].squeeze(0)
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"Observation {key}: expected {tuple(shape)}, got {tuple(value.shape)}.")
        result[key] = value
    return result


#########################
# video recording
#########################

class PlaybackVideoRecorder(gym.wrappers.RecordVideo):
    """Make synchronous video encoding visible in headless playback logs."""

    def stop_recording(self):
        frames = len(self.recorded_frames)
        started = time.perf_counter()
        if frames:
            path = Path(self.video_folder) / f"{self._video_name}.mp4"
            print(f"[INFO] Saving video: {path} ({frames} frames)...", flush=True)
        super().stop_recording()
        if frames:
            print(f"[INFO] Video saved in {time.perf_counter() - started:.1f}s: {path}", flush=True)


#########################
# rollout
#########################

@torch.inference_mode()
def rollout(policy, env, success_term, horizon, device):
    """Return success and episode length, stepping through the video wrapper when enabled."""
    base_env = env.unwrapped
    reset_started = time.perf_counter()
    print("[INFO] Resetting environment...", flush=True)
    policy.start_episode()
    obs_dict, _ = env.reset()
    print(f"[INFO] Environment reset complete ({time.perf_counter() - reset_started:.2f}s)", flush=True)
    rollout_started = time.perf_counter()

    for step in range(horizon):
        start_time = time.time()
        obs = policy_observations(obs_dict["policy"], policy.policy.obs_shapes)
        actions = policy(obs)
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min) / 2
            actions += args_cli.norm_factor_min
        actions = torch.from_numpy(actions).to(device=device).view(1, base_env.action_space.shape[1])
        obs_dict, _, terminated, truncated, _ = env.step(actions)

        sleep_time = base_env.step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        if step == 0 or (step + 1) % 100 == 0:
            print(f"[INFO] Rollout step={step + 1}/{horizon}, "
                  f"elapsed={time.perf_counter() - rollout_started:.1f}s", flush=True)

        if bool(success_term.func(base_env, **success_term.params)[0]):
            return True, step + 1
        if terminated or truncated:
            return False, step + 1

    return False, horizon


#########################
# seeded evaluation
#########################

def main():
    """Evaluate independent seeds and summarize success rates and episode lengths."""
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    log_dir = str(checkpoint_path.parent)
    video_dir = Path(args_cli.video_dir).expanduser().resolve()
    device = torch.device(args_cli.device) if args_cli.device is not None else TorchUtils.get_torch_device()
    policy = load_policy(checkpoint_path, device)
    success_rates = []
    episode_lengths = []
    total_successes = 0

    for seed_index, seed in enumerate(evaluation_seeds):
        if seed_index:
            create_new_stage()
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1,
                                use_fabric=not args_cli.disable_fabric)
        env_cfg.seed = seed
        env_cfg.log_dir = log_dir
        env_cfg.observations.policy.concatenate_terms = False
        env_cfg.terminations.time_out = None
        env_cfg.recorders = None
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None).unwrapped
        try:
            if args_cli.video:
                video_kwargs = {
                    "video_folder": str(video_dir),
                    "episode_trigger": lambda episode_id: True,
                    "video_length": args_cli.video_length,
                    "name_prefix": f"{checkpoint_path.stem}-seed-{seed}",
                    "disable_logger": True,
                }
                print("[INFO] Recording a separate video for each rollout.")
                print_dict(video_kwargs, nesting=4)
                env = PlaybackVideoRecorder(env, **video_kwargs)
            env.unwrapped.seed(seed)
            results = []
            print(f"\n[INFO] Evaluation seed={seed}, rollouts={args_cli.num_rollouts}")
            for trial in range(args_cli.num_rollouts):
                print(f"[INFO] Starting trial {trial} (seed={seed})", flush=True)
                success, steps = rollout(policy, env, success_term, args_cli.horizon, device)
                if args_cli.video and env.recording:
                    env.stop_recording()
                results.append(success)
                episode_lengths.append(steps)
                print(f"[INFO] Trial {trial}: {success}, steps={steps}\n", flush=True)

            successes = results.count(True)
            success_rates.append(successes / len(results))
            total_successes += successes
            print(f"[Seed {seed}]")
            print(f"Successful trials: {successes}, out of {len(results)} trials")
            print(f"Success rate: {success_rates[-1]}")
            print(f"Trial Results: {results}\n")
        except Exception as error:
            print(f"[ERROR] BC playback failed ({type(error).__name__}): {error}", flush=True)
            raise
        finally:
            env.close()
            del env

    mean_rate = float(np.mean(success_rates))
    std_rate = float(np.std(success_rates, ddof=0))
    print(f"\nEvaluation seeds: {evaluation_seeds}")
    print(f"Rollouts per seed: {args_cli.num_rollouts}")
    print(f"Total successful trials: {total_successes}, out of {len(episode_lengths)} trials")
    print(f"Success rates by seed: {dict(zip(evaluation_seeds, success_rates))}")
    print(f"Success rate mean ± std (across seeds, ddof=0): {mean_rate:.6f} ± {std_rate:.6f}")
    print(f"Success rate percent: {mean_rate * 100:.2f}% ± {std_rate * 100:.2f} percentage points")
    print(f"Episode length mean ± std (all rollouts, steps, ddof=0): "
          f"{np.mean(episode_lengths):.2f} ± {np.std(episode_lengths, ddof=0):.2f}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
