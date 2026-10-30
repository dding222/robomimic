"""Play and evaluate standalone ACT checkpoints in IsaacLab Toy2Box."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


#########################
# simulator startup
#########################

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Agibot-Toy2Box-RmpFlow-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--horizon", type=int, default=800)
parser.add_argument("--num_rollouts", type=int, default=1)
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--disable_fabric", action="store_true")
ensemble_args = parser.add_mutually_exclusive_group()
ensemble_args.add_argument("--temporal_ensemble_coeff", type=float, default=None,
                           help="Override the checkpoint and enable temporal ensemble, e.g. 0.01.")
ensemble_args.add_argument("--disable_temporal_ensemble", action="store_true",
                           help="Disable temporal ensemble for this playback.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.horizon < 1 or args_cli.num_rollouts < 1:
    parser.error("--horizon and --num_rollouts must be positive")

import torch

checkpoint = torch.load(args_cli.checkpoint.expanduser(), map_location="cpu", weights_only=True)
if not all(key in checkpoint for key in ("model", "state_dict", "state_keys")):
    parser.error("Expected a scripts/act/train.py checkpoint; use scripts/robomimic/play.py for BC checkpoints.")
args_cli.enable_cameras = args_cli.enable_cameras or checkpoint["model"]["enable_cam"]
checkpoint.pop("optimizer", None)
simulation_app = AppLauncher(args_cli).app

import random

import gymnasium as gym
import numpy as np
from robomimic.utils import obs_utils as ObsUtils

import agibot_toy2box  # noqa: F401
from agibot_toy2box.cameras import CAMERAS, add_camera_observations, setup_camera_viewports
from isaaclab_tasks.utils import parse_env_cfg
from toy2box_act import ACTConfig, ACTPolicy
from toy2box_act.dataset import prepare_observations


#########################
# rollout
#########################

@torch.inference_mode()
def rollout(policy, env, state_keys, success_term):
    policy.reset()
    obs, _ = env.reset()
    for step in range(1, args_cli.horizon + 1):
        if not simulation_app.is_running() or env.sim.is_stopped():
            return False, step - 1
        batch = prepare_observations(obs["policy"], state_keys, policy.config.camera_keys, env.device)
        if batch["observation.state"].shape[-1] != policy.config.state_dim:
            raise ValueError("Environment observation dimensions do not match the ACT checkpoint.")
        action = policy.select_action(batch)
        if action.shape != (1, 7) or not torch.isfinite(action).all():
            raise ValueError("ACT must produce finite (1, 7) actions.")
        obs, _, terminated, truncated, _ = env.step(action)
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, step
        if bool(terminated[0]) or bool(truncated[0]):
            return False, step
    return False, args_cli.horizon


#########################
# evaluation
#########################

def main():
    if args_cli.task != "Agibot-Toy2Box-RmpFlow-v0":
        raise ValueError("This ACT player supports Agibot-Toy2Box-RmpFlow-v0.")
    model_config = dict(checkpoint["model"])
    if args_cli.temporal_ensemble_coeff is not None:
        model_config["temporal_ensemble_coeff"] = args_cli.temporal_ensemble_coeff
        model_config["n_action_steps"] = 1
    elif args_cli.disable_temporal_ensemble:
        model_config["temporal_ensemble_coeff"] = None
    config = ACTConfig.from_dict(model_config)
    policy = ACTPolicy(config, checkpoint.get("normalization")).to(args_cli.device)
    policy.load_state_dict(checkpoint.pop("state_dict"))
    policy.eval()
    state_keys = checkpoint["state_keys"]
    names_by_key = {spec["key"]: name for name, spec in CAMERAS.items()}
    if any(key not in names_by_key for key in config.camera_keys):
        raise ValueError("Checkpoint camera keys must be hand_eye_cam, top_cam, or front_cam.")
    sensor_names = [names_by_key[key] for key in config.camera_keys]
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": state_keys, "rgb": config.camera_keys}})
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1,
                            use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.recorders = None
    if sensor_names:
        add_camera_observations(env_cfg, sensor_names)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    views = []
    try:
        if not args_cli.headless:
            views = setup_camera_viewports(sensor_names)
        progress = (f"step={checkpoint['step']}" if "step" in checkpoint
                    else f"epoch={checkpoint.get('epoch', 'unknown')}")
        print(f"ACT {progress}, cameras={config.camera_keys}, "
              f"action_chunk={config.action_chunk}, n_action_steps={config.n_action_steps}, "
              f"temporal_ensemble={config.temporal_ensemble_coeff}", flush=True)
        results = []
        for trial in range(args_cli.num_rollouts):
            if not simulation_app.is_running():
                break
            success, steps = rollout(policy, env, state_keys, success_term)
            results.append(success)
            print(f"Trial {trial + 1}: success={success}, steps={steps}", flush=True)
        if results:
            print(f"Success rate: {sum(results)}/{len(results)} ({100 * sum(results) / len(results):.1f}%)", flush=True)
    finally:
        for view in views:
            view.destroy()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
