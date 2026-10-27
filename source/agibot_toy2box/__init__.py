"""Agibot Toy2Box environments for imitation learning."""

import gymnasium as gym


BASE_TASK_ID = "Agibot-Toy2Box-RmpFlow-v0"
MIMIC_TASK_ID = "Agibot-Toy2Box-RmpFlow-Mimic-v0"


gym.register(
    id=BASE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "agibot_toy2box.place_toy2box_env_cfg:RmpFlowAgibotPlaceToy2BoxEnvCfg"
        ),
        "robomimic_bc_cfg_entry_point": "agibot_toy2box.agents:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

gym.register(
    id=MIMIC_TASK_ID,
    entry_point="isaaclab_mimic.envs.pick_place_mimic_env:PickPlaceRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": "agibot_toy2box.mimic_env_cfg:RmpFlowAgibotPlaceToy2BoxMimicEnvCfg"
    },
    disable_env_checker=True,
)
