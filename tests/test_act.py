"""ACT integration checks using real robomimic sequence sampling."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from toy2box_act import ACTConfig, ACTPolicy
from toy2box_act.dataset import inspect_dataset, make_dataset, prepare_batch


#########################
# test fixtures
#########################

STATE_KEYS = ["joint_pos", "joint_vel", "toy_truck_positions", "toy_truck_orientations",
              "box_positions", "box_orientations", "eef_pos", "eef_quat", "gripper_pos"]


def write_dataset(path):
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.attrs["num_samples"] = 3
        demo.create_dataset("actions", data=np.arange(21, dtype=np.float32).reshape(3, 7))
        obs = demo.create_group("obs")
        for key in STATE_KEYS:
            obs.create_dataset(key, data=np.ones((3, 2), dtype=np.float32))
        for key in ("hand_eye_cam", "top_cam", "front_cam", "fourth_cam"):
            obs.create_dataset(key, data=np.full((3, 32, 32, 3), 128, dtype=np.uint8))


def tiny_config(state_dim, cameras=()):
    return ACTConfig(state_dim=state_dim, enable_cam=bool(cameras), camera_keys=list(cameras),
                     seq_length=4, n_action_steps=4, dim_model=32, n_heads=4,
                     dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
                     n_vae_encoder_layers=1, dropout=0.0)


#########################
# integration tests
#########################

class ACTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "demos.hdf5"
        write_dataset(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_state_only_padding_and_raw_actions(self):
        keys, cams, dim = inspect_dataset(self.path)
        dataset = make_dataset(self.path, keys, cams, seq_length=4)
        try:
            samples = list(DataLoader(dataset, batch_size=1, shuffle=False))
            batch = prepare_batch(samples[-1], keys, cams, "cpu")
            self.assertEqual(batch["action_is_pad"].tolist(), [[False, True, True, True]])
            self.assertTrue(torch.equal(batch["action"][0, 0], torch.arange(14, 21).float()))
            self.assertTrue(torch.equal(batch["action"][0, 0], batch["action"][0, -1]))
            config = tiny_config(dim)
            config.use_vae = False
            model = ACTPolicy(config)
            self.assertFalse(hasattr(model.model, "backbone"))
            self.assertFalse(hasattr(model.model, "encoder_img_feat_input_proj"))
            self.assertFalse(hasattr(model.model, "encoder_cam_feat_pos_embed"))
            model.model.action_head.weight.data.zero_()
            model.model.action_head.bias.data.fill_(2.5)
            loss, _ = model(batch)
            expected = (batch["action"][:, :1] - 2.5).abs().mean()
            torch.testing.assert_close(loss, expected)
            loss.backward()
            chunk = model.predict_action_chunk(batch)
            self.assertEqual(tuple(chunk.shape), (1, 4, 7))
            torch.testing.assert_close(chunk, torch.full_like(chunk, 2.5))
            for _ in range(5):
                self.assertEqual(tuple(model.select_action(batch).shape), (1, 7))
            model.reset()
            self.assertEqual(len(model._action_queue), 0)
            checkpoint = Path(self.temp.name) / "model.pt"
            torch.save({"model": asdict(config), "state_dict": model.state_dict(), "state_keys": keys}, checkpoint)
            restored, restored_keys = ACTPolicy.from_checkpoint(checkpoint)
            self.assertEqual(restored_keys, keys)
            torch.testing.assert_close(restored.predict_action_chunk(batch), chunk)
        finally:
            dataset.close_and_delete_hdf5_handle()

    def test_camera_training_one_to_three_views(self):
        for n in (1, 2, 3):
            cams = ["eye", "top", "front"][:n]
            keys, cams, dim = inspect_dataset(self.path, True, cams)
            dataset = make_dataset(self.path, keys, cams, seq_length=4)
            try:
                batch = prepare_batch(next(iter(DataLoader(dataset, batch_size=2))), keys, cams, "cpu")
                self.assertEqual(len(batch["observation.images"]), n)
                self.assertEqual(tuple(batch["observation.images"][0].shape), (2, 3, 32, 32))
                torch.testing.assert_close(batch["observation.images"][0], torch.full((2, 3, 32, 32), 128 / 255))
                model = ACTPolicy(tiny_config(dim, cams))
                loss, _ = model(batch)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(model.model.backbone["conv1"].weight.grad)
                self.assertEqual(tuple(model.predict_action_chunk(batch).shape), (2, 4, 7))
            finally:
                dataset.close_and_delete_hdf5_handle()

    def test_invalid_camera_selection(self):
        with self.assertRaises(ValueError):
            inspect_dataset(self.path, True)  # Four automatically discovered cameras.
        with self.assertRaises(ValueError):
            inspect_dataset(self.path, True, ["missing_camera"])
        with self.assertRaises(ValueError):
            inspect_dataset(self.path, False, ["top_cam"])


if __name__ == "__main__":
    unittest.main()
