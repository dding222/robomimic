"""Compare ACT action predictions against recorded demonstrations without Isaac Sim."""

import argparse
import json
from pathlib import Path

import h5py
import torch
from robomimic.utils import obs_utils as ObsUtils

from toy2box_act import ACTConfig, ACTPolicy
from toy2box_act.dataset import prepare_observations
from toy2box_act.modeling_act import ACTTemporalEnsembler


#########################
# command line
#########################

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--num_demos", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.num_demos < 1 or args.batch_size < 1:
        parser.error("--num_demos and --batch_size must be positive")
    return args


#########################
# action error metrics
#########################

def transition_mask(actions):
    closed = actions[:, 6] < 0
    mask = torch.zeros(len(actions), dtype=torch.bool)
    for index in torch.nonzero(closed[1:] != closed[:-1]).flatten().tolist():
        step = index + 1
        mask[max(0, step - 5):min(len(actions), step + 6)] = True
    return mask


def error_metrics(predicted, target, transitions):
    error = predicted - target
    correct = (predicted[:, 6] < 0) == (target[:, 6] < 0)
    return {
        "samples": len(target),
        "position_mae_mm": (error[:, :3].abs().mean(0) * 1000).tolist(),
        "position_l2_mean_mm": (error[:, :3].norm(dim=1).mean() * 1000).item(),
        "rotation_vector_mae_deg": torch.rad2deg(error[:, 3:6].abs().mean(0)).tolist(),
        "gripper_sign_accuracy_pct": (correct.float().mean() * 100).item(),
        "gripper_transition_accuracy_pct": ((correct[transitions].float().mean() * 100).item()
                                            if transitions.any() else None),
    }


#########################
# recorded observation evaluation
#########################

@torch.inference_mode()
def main(args):
    checkpoint = torch.load(args.checkpoint.expanduser(), map_location="cpu", weights_only=True, mmap=True)
    config = ACTConfig.from_dict(checkpoint["model"])
    state_keys = checkpoint["state_keys"]
    path = (args.dataset or Path(checkpoint["dataset"])).expanduser()
    policy = ACTPolicy(config, checkpoint.get("normalization"))
    policy.load_state_dict(checkpoint["state_dict"])
    policy.to(args.device).eval()
    del checkpoint
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": state_keys, "rgb": config.camera_keys}})
    collected = {name: {key: [] for key in ("predicted", "target", "transitions")}
                 for name in ("first_action", "temporal_ensemble", "last_chunk_action")}
    print(f"Dataset={path}, action_chunk={config.action_chunk}, "
          f"temporal_ensemble={config.temporal_ensemble_coeff}, device={args.device}", flush=True)
    with h5py.File(path, "r") as handle:
        names = sorted(handle["data"], key=lambda name: int(name.rsplit("_", 1)[1]))
        if args.num_demos > len(names):
            raise ValueError(f"Requested {args.num_demos} demos; dataset has {len(names)}.")
        for demo_name in names[:args.num_demos]:
            demo = handle["data"][demo_name]
            target = torch.as_tensor(demo["actions"][()], dtype=torch.float32)
            chunks = []
            for start in range(0, len(target), args.batch_size):
                end = min(start + args.batch_size, len(target))
                obs = {key: demo["obs"][key][start:end] for key in state_keys + config.camera_keys}
                batch = prepare_observations(obs, state_keys, config.camera_keys, args.device)
                prediction = policy.predict_action_chunk(batch).cpu()
                if not torch.isfinite(prediction).all():
                    raise ValueError(f"{demo_name}: nonfinite prediction at observation {start}.")
                chunks.append(prediction)
            chunks = torch.cat(chunks)
            transitions = transition_mask(target)
            predictions = {"first_action": chunks[:, 0]}
            if config.temporal_ensemble_coeff is not None:
                ensemble = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.action_chunk)
                predictions["temporal_ensemble"] = torch.cat([
                    ensemble.update(chunk.unsqueeze(0)) for chunk in chunks
                ])
            report = {"demo": demo_name}
            for name, prediction in predictions.items():
                report[name] = error_metrics(prediction, target, transitions)
                collected[name]["predicted"].append(prediction)
                collected[name]["target"].append(target)
                collected[name]["transitions"].append(transitions)
            valid = len(target) - config.action_chunk + 1
            if valid > 0:
                prediction = chunks[:valid, -1]
                future = target[config.action_chunk - 1:]
                future_transitions = transitions[config.action_chunk - 1:]
                report["last_chunk_action"] = error_metrics(prediction, future, future_transitions)
                collected["last_chunk_action"]["predicted"].append(prediction)
                collected["last_chunk_action"]["target"].append(future)
                collected["last_chunk_action"]["transitions"].append(future_transitions)
            print(json.dumps(report), flush=True)
    overall = {
        name: error_metrics(torch.cat(values["predicted"]), torch.cat(values["target"]),
                            torch.cat(values["transitions"]))
        for name, values in collected.items() if values["predicted"]
    }
    print("Overall recorded-observation action errors:")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main(parse_args())
