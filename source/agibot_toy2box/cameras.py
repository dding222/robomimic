"""Shared camera poses for recording sensors and operator viewports."""

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import CameraCfg


#########################
# camera mounts
#########################

# All rotations use USD/OpenGL axes: forward -Z, image up +Y, quaternion wxyz.
CAMERAS = {
    "eye": {
        "key": "hand_eye_cam",
        "path": "/Robot/right_base_link/HandEyeCamera",
        "pos": (-0.0006198, -0.0891830, 0.0860144),
        "rot": (0.2233045, 0.9747158, 0.0038790, -0.0070149),
    },
    "top": {
        "key": "top_cam",
        "path": "/TopCamera",
        "pos": (0.3, -0.05, 0.75),
        "rot": (0.7071067812, 0.0, 0.0, -0.7071067812),
    },
    "front": {
        "key": "front_cam",
        "path": "/Robot/link_up_down_body/FrontCamera",
        "pos": (0.26, 0.0, -0.1),
        "rot": (0.5518671994, 0.4420888986, -0.4420888986, -0.5518671994),
    },
}


#########################
# camera creation
#########################

def camera_spawn_cfg():
    return sim_utils.PinholeCameraCfg(
        focal_length=18.0, focus_distance=1.0, horizontal_aperture=20.955,
        vertical_aperture=20.955 * 480 / 640, clipping_range=(0.02, 5.0),
    )


def add_camera_observations(env_cfg, names):
    if not names or len(set(names)) != len(names) or any(name not in CAMERAS for name in names):
        raise ValueError("Select unique camera names from eye, top, front.")
    env_cfg.image_obs_list = []
    for name in names:
        spec = CAMERAS[name]
        key = spec["key"]
        setattr(env_cfg.scene, key, CameraCfg(
            prim_path="{ENV_REGEX_NS}" + spec["path"],
            height=480, width=640, update_period=0.0, data_types=["rgb"],
            spawn=camera_spawn_cfg(),
            offset=CameraCfg.OffsetCfg(pos=spec["pos"], rot=spec["rot"], convention="opengl"),
        ))
        setattr(env_cfg.observations.policy, key, ObservationTermCfg(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg(key), "data_type": "rgb", "normalize": False},
        ))
        env_cfg.image_obs_list.append(key)


def spawn_view_cameras(names):
    for name in names:
        spec = CAMERAS[name]
        cfg = camera_spawn_cfg()
        cfg.func("/World/envs/env_0" + spec["path"], cfg,
                 translation=spec["pos"], orientation=spec["rot"])
