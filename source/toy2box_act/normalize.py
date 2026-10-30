"""Match the observation and action normalization used by local robomimic BC-RNN."""

import torch
from torch import nn


#########################
# observation normalization
#########################

class ObservationNormalizer(nn.Module):
    def __init__(self, stats):
        super().__init__()
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
        std = torch.as_tensor(stats["std"], dtype=torch.float32)
        if (mean.shape != std.shape or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("Observation statistics require finite means and positive standard deviations.")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, values):
        # SequenceDataset already adds robomimic's 1e-3 to std.
        return (values - self.mean) / self.std


#########################
# policy normalization
#########################

class ACTNormalizer(nn.Module):
    def __init__(self, config, stats=None):
        super().__init__()
        self.config = config
        self.images = nn.ModuleDict()
        if config.hdf5_normalize_obs:
            if stats is None or "state" not in stats:
                raise ValueError("Observation normalization requires training statistics from the checkpoint.")
            self.state = ObservationNormalizer(stats["state"])
            if self.state.mean.shape != (config.state_dim,):
                raise ValueError("State normalization statistics do not match state_dim.")
            for key in config.camera_keys:
                self.images[key] = ObservationNormalizer(stats["images"][key])
        if config.normalize_training_actions:
            if stats is None or "action" not in stats:
                raise ValueError("Action normalization requires training min/max from the checkpoint.")
            action_min = torch.as_tensor(stats["action"]["min"], dtype=torch.float32)
            action_max = torch.as_tensor(stats["action"]["max"], dtype=torch.float32)
            if (action_min.numel() != 1 or action_max.numel() != 1
                    or not torch.isfinite(action_min).all() or not torch.isfinite(action_max).all()
                    or not (action_max > action_min).all()):
                raise ValueError("Action normalization requires finite dataset-wide scalar min < max.")
            self.register_buffer("action_min", action_min)
            self.register_buffer("action_max", action_max)

    def observations(self, batch):
        batch = dict(batch)
        if self.config.hdf5_normalize_obs:
            batch["observation.state"] = self.state(batch["observation.state"])
            if self.config.camera_keys:
                batch["observation.images"] = [
                    self.images[key](image)
                    for key, image in zip(self.config.camera_keys, batch["observation.images"])
                ]
        return batch

    def targets(self, actions):
        if self.config.normalize_training_actions:
            return 2 * ((actions - self.action_min) / (self.action_max - self.action_min)) - 1
        return actions

    def restore(self, actions):
        if self.config.normalize_training_actions:
            return ((actions + 1) * (self.action_max - self.action_min)) / 2 + self.action_min
        return actions
