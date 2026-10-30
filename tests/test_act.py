"""ACT integration checks using real robomimic sequence sampling."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from robomimic.utils.dataset import SequenceDataset

from toy2box_act import ACTConfig, ACTPolicy
from toy2box_act.dataset import (
    batch_to_device, collate_batch, inspect_dataset, make_dataset, normalization_stats, prepare_batch,
)
from toy2box_act.modeling_act import ACTTemporalEnsembler


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
                     action_chunk=4, n_action_steps=4, dim_model=32, n_heads=4,
                     dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
                     n_vae_encoder_layers=1, dropout=0.0, pretrained_backbone_weights=None)


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
        dataset = make_dataset(self.path, keys, cams, action_chunk=4)
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
            expected = (batch["action"][:, 0] - 2.5).abs().mean() / 4
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

    def test_current_observation_adapter_matches_rnn_sequences(self):
        with h5py.File(self.path, "r+") as handle:
            handle["data/demo_0/obs/joint_pos"][:] = np.arange(6, dtype=np.float32).reshape(3, 2)
        keys, cams, _ = inspect_dataset(self.path, True, ["top"])
        dataset = make_dataset(self.path, keys, cams, action_chunk=4, hdf5_normalize_obs=False)
        baseline = SequenceDataset(
            hdf5_path=str(self.path), obs_keys=keys + cams, dataset_keys=["actions"],
            frame_stack=1, seq_length=4, pad_frame_stack=True, pad_seq_length=True,
            get_pad_mask=True, hdf5_cache_mode="low_dim", hdf5_use_swmr=True,
            hdf5_normalize_obs=False, load_next_obs=False,
        )
        try:
            samples, baseline_samples = [], []
            for index in range(len(dataset)):
                sample, original = dataset[index], baseline[index]
                for key in keys + cams:
                    self.assertEqual(sample["obs"][key].shape[0], 1)
                    np.testing.assert_array_equal(sample["obs"][key][0], original["obs"][key][0])
                np.testing.assert_array_equal(sample["actions"], original["actions"])
                np.testing.assert_array_equal(sample["pad_mask"], original["pad_mask"])
                samples.append(sample)
                baseline_samples.append(original)
            expected = prepare_batch(default_collate(baseline_samples), keys, cams, "cpu")
            actual = batch_to_device(collate_batch(samples, keys, cams), "cpu")
            for key in ("observation.state", "action", "action_is_pad"):
                torch.testing.assert_close(actual[key], expected[key])
            for image, reference in zip(actual["observation.images"], expected["observation.images"]):
                torch.testing.assert_close(image, reference)
        finally:
            dataset.close_and_delete_hdf5_handle()
            baseline.close_and_delete_hdf5_handle()

    def test_camera_training_one_to_three_views(self):
        for n in (1, 2, 3):
            cams = ["eye", "top", "front"][:n]
            keys, cams, dim = inspect_dataset(self.path, True, cams)
            dataset = make_dataset(self.path, keys, cams, action_chunk=4)
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

    def test_rnn_normalization_and_checkpoint_roundtrip(self):
        keys, cams, dim = inspect_dataset(self.path, True, ["top"])
        dataset = make_dataset(self.path, keys, cams, action_chunk=4, hdf5_normalize_obs=True)
        try:
            stats = normalization_stats(dataset, keys, cams, normalize_actions=True)
            sample = next(iter(DataLoader(dataset, batch_size=2)))
            batch = prepare_batch(sample, keys, cams, "cpu")
            config = tiny_config(dim, cams)
            config.hdf5_normalize_obs = True
            config.normalize_training_actions = True
            model = ACTPolicy(config, stats)
            normalized = model.normalizer.observations(batch)
            obs_stats = dataset.get_obs_normalization_stats()
            expected_state = torch.cat([
                ((sample["obs"][key][:, 0] - torch.as_tensor(obs_stats[key]["mean"][0])) /
                 torch.as_tensor(obs_stats[key]["std"][0])).flatten(start_dim=1)
                for key in keys
            ], dim=-1)
            torch.testing.assert_close(normalized["observation.state"], expected_state)
            for key, image, result in zip(cams, batch["observation.images"], normalized["observation.images"]):
                expected = ((image - torch.as_tensor(obs_stats[key]["mean"][0])) /
                            torch.as_tensor(obs_stats[key]["std"][0]))
                torch.testing.assert_close(result, expected)
            self.assertEqual(stats["action"], {"min": 0.0, "max": 20.0})
            normalized_actions = model.normalizer.targets(batch["action"])
            torch.testing.assert_close(normalized_actions, 2 * (batch["action"] / 20) - 1)
            torch.testing.assert_close(model.normalizer.restore(normalized_actions), batch["action"])
            # A normalized output of zero must become the environment-scale midpoint.
            model.model.action_head.weight.data.zero_()
            model.model.action_head.bias.data.zero_()
            torch.testing.assert_close(model.predict_action_chunk(batch), torch.full((2, 4, 7), 10.0))
            path = Path(self.temp.name) / "normalized.pt"
            torch.save({"model": asdict(config), "state_dict": model.state_dict(),
                        "state_keys": keys, "normalization": stats}, path)
            restored, _ = ACTPolicy.from_checkpoint(path)
            torch.testing.assert_close(restored.predict_action_chunk(batch), model.predict_action_chunk(batch))
            with self.assertRaises(ValueError):
                ACTPolicy(config)
        finally:
            dataset.close_and_delete_hdf5_handle()

    def test_legacy_checkpoint_keeps_raw_units_and_queue(self):
        config = tiny_config(18)
        model = ACTPolicy(config)
        legacy_config = asdict(config)
        legacy_config["seq_length"] = legacy_config.pop("action_chunk")
        for key in ("hdf5_normalize_obs", "normalize_training_actions", "temporal_ensemble_coeff"):
            legacy_config.pop(key)
        path = Path(self.temp.name) / "legacy.pt"
        torch.save({"model": legacy_config, "state_dict": model.state_dict(), "state_keys": STATE_KEYS}, path)
        restored, _ = ACTPolicy.from_checkpoint(path)
        self.assertEqual(restored.config.action_chunk, config.action_chunk)
        self.assertIsNone(restored.temporal_ensembler)
        self.assertFalse(restored.config.hdf5_normalize_obs)
        self.assertFalse(restored.config.normalize_training_actions)
        batch = {"observation.state": torch.ones(2, 18)}
        torch.testing.assert_close(restored.predict_action_chunk(batch), model.predict_action_chunk(batch))

    def test_temporal_ensemble_matches_explicit_overlap_average(self):
        for chunk_size in (1, 4):
            for coeff in (0.0, 0.01, -0.01):
                ensemble = ACTTemporalEnsembler(coeff, chunk_size)
                history = []
                for step in range(2 * chunk_size + 3):
                    chunk = torch.arange(2 * chunk_size * 7).reshape(2, chunk_size, 7).float()
                    chunk = chunk * (-1 if step % 2 else 1) + step
                    history.append(chunk)
                    first = max(0, step - chunk_size + 1)
                    overlapping = torch.stack([history[start][:, step - start] for start in range(first, step + 1)], dim=1)
                    weights = torch.exp(-coeff * torch.arange(step - first + 1))
                    expected = (overlapping * weights[None, :, None]).sum(1) / weights.sum()
                    torch.testing.assert_close(ensemble.update(chunk), expected)
                ensemble.reset()
                torch.testing.assert_close(ensemble.update(history[0]), history[0][:, 0])

    def test_temporal_policy_predicts_every_step_and_resets(self):
        config = tiny_config(18)
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = 0.01
        model = ACTPolicy(config)
        chunks = [torch.full((2, 4, 7), float(step)) for step in range(6)]
        with patch.object(model, "predict_action_chunk", side_effect=chunks) as predict:
            for _ in chunks:
                self.assertEqual(tuple(model.select_action({}).shape), (2, 7))
            self.assertEqual(predict.call_count, len(chunks))
        model.reset()
        self.assertIsNone(model.temporal_ensembler.ensembled_actions)
        self.assertEqual(len(model._action_queue), 0)
        with self.assertRaises(ValueError):
            ACTConfig(state_dim=18, temporal_ensemble_coeff=0.01, n_action_steps=10)


if __name__ == "__main__":
    unittest.main()
