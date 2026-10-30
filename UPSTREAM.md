# Upstream source

가져온 기준은 로컬 `/home/user/package/IsaacLab`의 아래 revision이다.

- Repository: `https://github.com/isaac-sim/IsaacLab.git`
- Tag: `v2.3.2`
- Commit: `37ddf626871758333d6ed89cf64ad702aef127d0`
- Commit date: `2026-01-29`

## Source mapping

| 이 저장소 | Isaac Lab v2.3.2 원본 |
|---|---|
| `source/agibot_toy2box/agibot.py` | `source/isaaclab_assets/isaaclab_assets/robots/agibot.py` |
| `source/agibot_toy2box/place_toy2box_env_cfg.py` | `source/isaaclab_tasks/.../place/config/agibot/place_toy2box_rmp_rel_env_cfg.py` |
| `source/agibot_toy2box/mimic_env_cfg.py` | `source/isaaclab_mimic/isaaclab_mimic/envs/agibot_place_toy2box_mimic_env_cfg.py` |
| `source/isaaclab_mimic/` | `source/isaaclab_mimic/isaaclab_mimic/` 전체 Python source |
| `scripts/teleoperation/` | `scripts/environments/teleoperation/teleop_se3_agent.py`, `scripts/tools/record_demos.py`, `scripts/tools/replay_demos.py` |
| `scripts/isaaclab_mimic/` | `scripts/imitation_learning/isaaclab_mimic/` |
| `scripts/robomimic/` | `scripts/imitation_learning/robomimic/` |
| `docs/` | Isaac Lab imitation-learning 문서와 Isaac Lab Mimic changelog |

로컬 패키지로 독립 등록하기 위해 import 경로와 Gym task ID만 변경했고, 복사한 upstream 파일의
저작권 및 SPDX header는 유지했다. 프로젝트 전체 라이선스 사본은 `LICENSE`에 있다.

## ACT

- Source: `/home/user/lerobot/src/lerobot/policies/act/modeling_act.py` and `configuration_act.py`
- Local repository HEAD: `75ededdd660f5fc46ab0de59ff5e4fa6f466df75` (local source modifications included)
- Original authors: Tony Z. Zhao and The HuggingFace Inc. team, copyright 2024
- License: Apache-2.0 (headers retained in the adapted model and configuration)
- License copy: `source/toy2box_act/LICENSE.Apache-2.0`
- Destination: `source/toy2box_act/`
- Adaptations: independent PyTorch policy/configuration, optional 0–3 cameras, state-only forward,
  raw 7-dimensional Toy2Box action output and robomimic SequenceDataset adapter.
- Transformer/CVAE/ResNet computation, padded masked L1 mean, KL loss and temporal ensemble follow the local ACT source.
  The camera backbone uses the original ResNet18 ImageNet initialization; state-only training creates no backbone.
- The local LeRobot source includes custom debug-output changes; these are omitted from this adapter.
- Temporal ensemble: `ACTTemporalEnsembler` copied unchanged from that local ACT source.
  New training defaults to coefficient 0.01 and one action per observation; legacy checkpoints keep their previous behavior.
- Normalization: matches local robomimic BC-RNN options, using SequenceDataset observation statistics
  and the BC training script's global scalar action min/max. Defaults follow BC-RNN (observation normalization off,
  action normalization opt-in). Statistics are saved in ACT checkpoints and restored at inference.
- Training follows LeRobot's TF32, cuDNN benchmark, optional AMP/GradScaler and AdamW update settings.
  The adapter prepares current observations on CPU before DataLoader memory pinning and asynchronous CUDA transfer.
  Action sampling, episode boundaries and padding still use robomimic SequenceDataset.
  Training uses LeRobot's step-based schedule: 100,000 total updates, logging every 200 steps,
  and checkpointing every 1,000 steps plus the last step. The DataLoader cycle follows the local LeRobot implementation.
