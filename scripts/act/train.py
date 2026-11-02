"""Train optional-camera ACT on the same HDF5 sequences as robomimic BC-RNN."""

import argparse
import csv
import json
import random
import shutil
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
import numpy as np
from torch.utils.data import DataLoader

from toy2box_act import ACTConfig, ACTPolicy
from toy2box_act.dataset import (
    batch_to_device, collate_batch, inspect_dataset, make_dataset, normalization_stats, rnn_train_config,
)


#########################
# command line
#########################

def parse_bool(value):
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError("Use True or False.")


def parse_ensemble_coeff(value):
    return None if value.lower() == "none" else float(value)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="HDF5 dataset; defaults to the checkpoint dataset when resuming.")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Default: logs/act, or a resume_step_<step> folder beside the checkpoint.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume an ACT step checkpoint.")
    parser.add_argument("--enable_cam", type=parse_bool, default=False)
    parser.add_argument("--cameras", "--camera_keys", dest="camera_keys", nargs="+", default=None,
                        help="Select eye, top, front. --camera_keys also accepts legacy HDF5 key names.")
    parser.add_argument("--action_chunk", type=int, default=32)
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--temporal_ensemble_coeff", type=parse_ensemble_coeff, default=0.01,
                        help="ACT temporal ensemble coefficient; default 0.01. Use None to disable.")
    parser.add_argument("--norm", dest="hdf5_normalize_obs", type=parse_bool, default=None,
                        help="Normalize observations with training mean/std. Default: BC-RNN JSON (False), or checkpoint on resume.")
    parser.add_argument("--normalize_training_actions", action="store_true",
                        help="Use BC-RNN's dataset-wide scalar min/max action normalization.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Final total step, not additional steps. Default: 100000 or checkpoint target.")
    parser.add_argument("--log_freq", type=int, default=200)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--use_amp", type=parse_bool, default=False,
                        help="Use LeRobot's optional automatic mixed precision; default False.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if args.dataset is None and args.resume is None:
        parser.error("--dataset is required unless --resume is provided")
    return args


#########################
# training
#########################

def capture_rng_state(device):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng_state(state, device):
    random.setstate(state["python"])
    algorithm, values, position, has_gauss, cached = state["numpy"]
    np.random.set_state((algorithm, np.asarray(values, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def train(args):
    checkpoint = None
    start_step = 0
    saved_train = {}
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        required = {"model", "state_dict", "optimizer", "step", "state_keys", "dataset"}
        if not required.issubset(checkpoint):
            raise ValueError("--resume requires an ACT step checkpoint with optimizer state.")
        start_step = checkpoint["step"]
        saved_train = checkpoint.get("train", {})
        for key in ("batch_size", "num_workers", "lr", "seed", "use_amp", "log_freq", "save_freq"):
            if key in saved_train:
                setattr(args, key, saved_train[key])
        if args.dataset is None:
            args.dataset = Path(checkpoint["dataset"])
    if args.steps is None:
        args.steps = saved_train.get("steps", 100_000)
    if args.output_dir is None:
        args.output_dir = (args.resume.parent / f"resume_step_{start_step}"
                           if args.resume is not None else Path("logs/act"))
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.steps < 1 or args.batch_size < 1 or args.save_freq < 1:
        raise ValueError("steps, batch_size and save_freq must be positive.")
    if args.num_workers < 0 or args.log_freq < 0:
        raise ValueError("num_workers and log_freq must be nonnegative.")
    if args.steps <= start_step:
        raise ValueError(f"--steps must exceed checkpoint step {start_step}.")
    if (args.output_dir / "metrics.csv").exists():
        raise FileExistsError("metrics.csv already exists; use a new --output_dir to preserve the previous run.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    path = args.dataset.expanduser().resolve()
    if checkpoint is None:
        state_keys, camera_keys, state_dim = inspect_dataset(path, args.enable_cam, args.camera_keys)
        config = ACTConfig(
            state_dim=state_dim,
            enable_cam=args.enable_cam,
            camera_keys=camera_keys,
            action_chunk=args.action_chunk,
            n_action_steps=(args.n_action_steps if args.n_action_steps is not None else
                            (1 if args.temporal_ensemble_coeff is not None else args.action_chunk)),
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            hdf5_normalize_obs=(args.hdf5_normalize_obs if args.hdf5_normalize_obs is not None else
                                rnn_train_config()["hdf5_normalize_obs"]),
            normalize_training_actions=args.normalize_training_actions,
        )
    else:
        config = ACTConfig.from_dict(checkpoint["model"])
        if args.hdf5_normalize_obs is not None and args.hdf5_normalize_obs != config.hdf5_normalize_obs:
            raise ValueError("Observation normalization must match the resumed checkpoint. Start a new run to change it.")
        state_keys, camera_keys, state_dim = inspect_dataset(
            path, config.enable_cam, config.camera_keys, state_keys=checkpoint["state_keys"],
        )
        if state_keys != checkpoint["state_keys"] or state_dim != config.state_dim:
            raise ValueError("Dataset observations do not match the resumed ACT checkpoint.")
    dataset = make_dataset(
        path, state_keys, camera_keys, config.action_chunk,
        config.hdf5_normalize_obs if checkpoint is None else False,
    )
    stats = (normalization_stats(dataset, state_keys, camera_keys, config.normalize_training_actions)
             if checkpoint is None else checkpoint.get("normalization"))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=partial(collate_batch, state_keys=state_keys, camera_keys=camera_keys),
        pin_memory=device.type == "cuda", prefetch_factor=2 if args.num_workers > 0 else None,
    )
    model = ACTPolicy(config, stats).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    grad_scaler = torch.amp.GradScaler(device.type, enabled=args.use_amp)
    if checkpoint is not None:
        model.load_state_dict(checkpoint.pop("state_dict"))
        optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if args.use_amp and not checkpoint.get("grad_scaler"):
            raise ValueError("AMP resume requires GradScaler state in the checkpoint.")
        if "grad_scaler" in checkpoint:
            grad_scaler.load_state_dict(checkpoint["grad_scaler"])
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"], device)
    optimizer.zero_grad()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": asdict(config), "state_keys": state_keys, "dataset": str(path),
        "train": {
            "steps": args.steps, "log_freq": args.log_freq, "save_freq": args.save_freq,
            "batch_size": args.batch_size, "num_workers": args.num_workers,
            "lr": args.lr, "seed": args.seed, "use_amp": args.use_amp, "device": str(device),
        },
    }
    if args.resume is not None:
        run_config["resume_from"] = str(args.resume)
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    print(f"state_dim={state_dim}, action_dim=7, action_chunk={config.action_chunk}, cameras={camera_keys}")
    print(f"state_keys={state_keys}")
    print(f"Normalization: observations={config.hdf5_normalize_obs}, actions={config.normalize_training_actions}")
    print(f"Temporal ensemble={config.temporal_ensemble_coeff}, n_action_steps={config.n_action_steps}")
    print(f"device={device}, batch_size={args.batch_size}, num_workers={args.num_workers}, "
          f"use_amp={args.use_amp}, steps={args.steps}, log_freq={args.log_freq}, save_freq={args.save_freq}")
    if checkpoint is not None:
        print(f"Resuming step {start_step}; next step={start_step + 1}, output_dir={args.output_dir}")
    try:
        with (args.output_dir / "metrics.csv").open("x", newline="") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=["step", "loss", "l1", "kl", "data_s", "update_s"])
            writer.writeheader()
            metrics_file.flush()
            data_iter = cycle(loader)
            model.train()
            totals = {"loss": 0.0, "l1": 0.0, "kl": 0.0, "data_s": 0.0, "update_s": 0.0}
            count = 0
            for step in range(start_step + 1, args.steps + 1):
                started = perf_counter()
                sample = next(data_iter)
                data_s = perf_counter() - started
                batch = batch_to_device(sample, device)
                started = perf_counter()
                with torch.autocast(device_type=device.type) if args.use_amp else nullcontext():
                    loss, metrics = model(batch)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=False)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
                metrics["data_s"] = data_s
                metrics["update_s"] = perf_counter() - started
                for key in totals:
                    totals[key] += metrics[key]
                count += 1
                is_log_step = (args.log_freq > 0 and step % args.log_freq == 0) or step == args.steps
                is_saving_step = step % args.save_freq == 0 or step == args.steps
                if is_log_step or is_saving_step:
                    averages = {key: value / count for key, value in totals.items()}
                if is_log_step:
                    print(f"step={step} " + " ".join(f"{k}={v:.6f}" for k, v in averages.items()), flush=True)
                    writer.writerow({"step": step, **averages})
                    metrics_file.flush()
                    totals = {key: 0.0 for key in totals}
                    count = 0
                if is_saving_step:
                    checkpoint = {
                        **run_config,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "grad_scaler": grad_scaler.state_dict(),
                        "step": step,
                        "metrics": averages,
                        "normalization": stats,
                        "rng_state": capture_rng_state(device),
                    }
                    checkpoint_path = args.output_dir / f"model_step_{step}.pt"
                    temporary = checkpoint_path.with_suffix(".tmp")
                    torch.save(checkpoint, temporary)
                    temporary.replace(checkpoint_path)
                    latest_temporary = args.output_dir / "latest.tmp"
                    shutil.copyfile(checkpoint_path, latest_temporary)
                    latest_temporary.replace(args.output_dir / "latest.pt")
    finally:
        dataset.close_and_delete_hdf5_handle()


if __name__ == "__main__":
    train(parse_args())
