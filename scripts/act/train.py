"""Train optional-camera ACT on the same HDF5 sequences as robomimic BC-RNN."""

import argparse
import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from toy2box_act import ACTConfig, ACTPolicy
from toy2box_act.dataset import inspect_dataset, make_dataset, prepare_batch


#########################
# command line
#########################

def parse_bool(value):
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError("Use True or False.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("logs/act"))
    parser.add_argument("--enable_cam", type=parse_bool, default=False)
    parser.add_argument("--cameras", "--camera_keys", dest="camera_keys", nargs="+", default=None,
                        help="Select eye, top, front. --camera_keys also accepts legacy HDF5 key names.")
    parser.add_argument("--seq_length", type=int, default=10)
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


#########################
# training
#########################

def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs and batch_size must be positive; num_workers must be nonnegative.")
    if (args.output_dir / "metrics.csv").exists():
        raise FileExistsError("metrics.csv already exists; use a new --output_dir to preserve the previous run.")
    torch.manual_seed(args.seed)
    path = args.dataset.expanduser().resolve()
    state_keys, camera_keys, state_dim = inspect_dataset(path, args.enable_cam, args.camera_keys)
    config = ACTConfig(
        state_dim=state_dim,
        enable_cam=args.enable_cam,
        camera_keys=camera_keys,
        seq_length=args.seq_length,
        n_action_steps=args.n_action_steps if args.n_action_steps is not None else args.seq_length,
    )
    dataset = make_dataset(path, state_keys, camera_keys, config.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = ACTPolicy(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"model": asdict(config), "state_keys": state_keys, "dataset": str(path)}
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    print(f"state_dim={state_dim}, action_dim=7, seq_length={config.seq_length}, cameras={camera_keys}")
    print("Action targets: original HDF5 actions, without rescaling.")
    try:
        with (args.output_dir / "metrics.csv").open("x", newline="") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=["epoch", "loss", "l1", "kl"])
            writer.writeheader()
            metrics_file.flush()
            for epoch in range(1, args.epochs + 1):
                model.train()
                totals = {"loss": 0.0, "l1": 0.0, "kl": 0.0}
                count = 0
                for sample in loader:
                    batch = prepare_batch(sample, state_keys, camera_keys, args.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, metrics = model(batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite ACT loss; inspect the input dataset.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                    size = batch["action"].shape[0]
                    for key in totals:
                        totals[key] += metrics[key] * size
                    count += size
                metrics = {k: v / count for k, v in totals.items()}
                print(f"epoch={epoch} " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()), flush=True)
                writer.writerow({"epoch": epoch, **metrics})
                metrics_file.flush()
                if epoch % 100 == 0 or epoch == args.epochs:
                    checkpoint = {
                        **run_config,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "metrics": metrics,
                    }
                    checkpoint_path = args.output_dir / f"model_epoch_{epoch}.pt"
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
