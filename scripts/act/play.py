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
parser.add_argument("--check_observations", action="store_true",
                    help="Compare full episodes for --num_rollouts dataset demos without applying policy actions.")
observation_steps = parser.add_mutually_exclusive_group()
observation_steps.add_argument("--check_obs_step", type=int, default=None,
                               help="Compare only this observation index in each selected demo; default is all steps.")
observation_steps.add_argument("--check_obs_all_steps", action="store_true",
                               help="Compare every observation in each selected demo; enables --check_observations.")
ensemble_args = parser.add_mutually_exclusive_group()
ensemble_args.add_argument("--temporal_ensemble_coeff", type=float, default=None,
                           help="Override the checkpoint and enable temporal ensemble, e.g. 0.01.")
ensemble_args.add_argument("--disable_temporal_ensemble", action="store_true",
                           help="Disable temporal ensemble for this playback.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.check_observations = (args_cli.check_observations or args_cli.check_obs_all_steps
                               or args_cli.check_obs_step is not None)
if args_cli.horizon < 1 or args_cli.num_rollouts < 1:
    parser.error("--horizon and --num_rollouts must be positive")
if args_cli.check_obs_step is not None and args_cli.check_obs_step < 0:
    parser.error("--check_obs_step must be nonnegative")

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
# observation check
#########################

@torch.inference_mode()
def check_observations(policy, env, state_keys, demo_name):
    import h5py

    with h5py.File(checkpoint["dataset"], "r") as handle:
        demo = handle[f"data/{demo_name}"]
        length = len(demo["actions"])
        all_steps = args_cli.check_obs_step is None
        if not all_steps and args_cli.check_obs_step >= length:
            raise ValueError(f"--check_obs_step exceeds {demo_name} length ({length}).")
        steps = range(length) if all_steps else [args_cli.check_obs_step]
        maximum_errors = {key: 0.0 for key in state_keys}
        mismatch_counts = {key: 0 for key in state_keys}
        first_mismatches = {}
        input_error = 0.0
        input_mismatches = 0
        failed_steps = 0
        print(f"Observation comparison: {demo_name}, episode_steps={length}, "
              f"checking={len(steps)} observation(s), dataset={checkpoint['dataset']}", flush=True)
        for checked, step in enumerate(steps, 1):
            if not simulation_app.is_running():
                raise RuntimeError("Observation check interrupted before completion.")
            group = demo["initial_state"] if step == 0 else demo["states"]
            state_index = 0 if step == 0 else step - 1
            state = {
                kind: {
                    asset: {key: torch.as_tensor(value[state_index:state_index + 1], device=env.device)
                            for key, value in fields.items()}
                    for asset, fields in assets.items()
                }
                for kind, assets in group.items()
            }
            expected = {key: torch.as_tensor(demo["obs"][key][step:step + 1], device=env.device)
                        for key in state_keys}
            obs, _ = env.reset_to(state, env_ids=None, is_relative=True)
            actual = obs["policy"]
            matched = True
            for key in state_keys:
                if actual[key].shape != expected[key].shape:
                    raise ValueError(f"{key}: runtime shape {actual[key].shape}, dataset shape {expected[key].shape}.")
                error = (actual[key] - expected[key]).abs().max().item()
                if not np.isfinite(error):
                    error = float("inf")
                maximum_errors[key] = max(maximum_errors[key], error)
                if error > 1e-4:
                    matched = False
                    mismatch_counts[key] += 1
                    if key not in first_mismatches:
                        first_mismatches[key] = step
                        print(f"First mismatch: {demo_name}, {key}, step={step}, max_abs_diff={error:.8g}")
                        print(f"  dataset={expected[key].tolist()}\n  runtime={actual[key].tolist()}")
            actual_state = prepare_observations(actual, state_keys, [], env.device)["observation.state"]
            expected_state = prepare_observations(expected, state_keys, [], env.device)["observation.state"]
            if policy.config.hdf5_normalize_obs:
                actual_state = policy.normalizer.state(actual_state)
                expected_state = policy.normalizer.state(expected_state)
            error = (actual_state - expected_state).abs().max().item()
            if not np.isfinite(error):
                error = float("inf")
            input_error = max(input_error, error)
            input_mismatches += int(error > 1e-4)
            failed_steps += int(not matched or error > 1e-4)
            if all_steps and checked % 100 == 0:
                print(f"{demo_name}: checked {checked}/{len(steps)} observations", flush=True)
    for key in state_keys:
        print(f"{key}: max_abs_diff={maximum_errors[key]:.8g}, "
              f"mismatched_steps={mismatch_counts[key]}/{len(steps)}, "
              f"first_mismatch={first_mismatches.get(key, 'none')}")
    print(f"ACT state input: shape={tuple(actual_state.shape)}, max_abs_diff={input_error:.8g}, "
          f"mismatched_steps={input_mismatches}/{len(steps)}")
    print(f"Observation check: {'MATCHED' if failed_steps == 0 else 'MISMATCH'}, "
          f"matched_steps={len(steps) - failed_steps}/{len(steps)} (tolerance=1e-4). RGB is not compared.")
    return {
        "demo": demo_name, "matched": failed_steps == 0,
        "matched_steps": len(steps) - failed_steps, "checked_steps": len(steps),
    }


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
        if args_cli.check_observations:
            import h5py

            with h5py.File(checkpoint["dataset"], "r") as handle:
                demo_names = sorted(handle["data"], key=lambda name: int(name.rsplit("_", 1)[1]))
            if args_cli.num_rollouts > len(demo_names):
                raise ValueError(f"Requested {args_cli.num_rollouts} observation checks, "
                                 f"but the dataset has {len(demo_names)} demos.")
            results = []
            for trial, demo_name in enumerate(demo_names[:args_cli.num_rollouts], 1):
                print(f"Starting observation check {trial}/{args_cli.num_rollouts}: {demo_name}", flush=True)
                result = check_observations(policy, env, state_keys, demo_name)
                results.append(result)
                print(f"Trial {trial}/{args_cli.num_rollouts}: demo={demo_name}, "
                      f"observation_check={'MATCHED' if result['matched'] else 'MISMATCH'}, "
                      f"matched_steps={result['matched_steps']}/{result['checked_steps']}", flush=True)
            matched = sum(result["matched"] for result in results)
            matched_steps = sum(result["matched_steps"] for result in results)
            checked_steps = sum(result["checked_steps"] for result in results)
            print(f"Observation summary: matched_episodes={matched}/{len(results)}, "
                  f"matched_steps={matched_steps}/{checked_steps}", flush=True)
            return
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
