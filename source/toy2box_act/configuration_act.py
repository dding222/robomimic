# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACT configuration adapted from the local LeRobot implementation."""

from dataclasses import dataclass, field


#########################
# model configuration
#########################

@dataclass
class Feature:
    shape: tuple[int, ...]


@dataclass
class ACTConfig:
    state_dim: int
    action_dim: int = 7
    enable_cam: bool = False
    camera_keys: list[str] = field(default_factory=list)
    seq_length: int = 10
    n_action_steps: int = 10
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None
    replace_final_stride_with_dilation: bool = False
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    dropout: float = 0.1
    kl_weight: float = 10.0

    def __post_init__(self):
        if self.state_dim <= 0 or self.action_dim != 7:
            raise ValueError("Toy2Box requires state_dim > 0 and 7 robomimic actions.")
        if self.enable_cam and not 1 <= len(self.camera_keys) <= 3:
            raise ValueError("--enable_cam True requires 1 to 3 camera keys.")
        if not self.enable_cam and self.camera_keys:
            raise ValueError("Camera keys require --enable_cam True.")
        if len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("Camera keys must be unique.")
        if not 1 <= self.n_action_steps <= self.seq_length:
            raise ValueError("Require 1 <= n_action_steps <= seq_length.")
        if self.dim_model % 4 or self.dim_model % self.n_heads:
            raise ValueError("dim_model must be divisible by 4 and n_heads.")

    @property
    def chunk_size(self):
        return self.seq_length

    @property
    def robot_state_feature(self):
        return Feature((self.state_dim,))

    @property
    def action_feature(self):
        return Feature((self.action_dim,))

    @property
    def image_features(self):
        return self.camera_keys if self.enable_cam else []

    @property
    def env_state_feature(self):
        return None
