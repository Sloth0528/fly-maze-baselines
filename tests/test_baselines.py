"""Behavior tests; any test training stays inside a temporary directory."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from maze_baselines.environment import ContinuousMaze
from maze_baselines.experiment import evaluate, train
from maze_baselines.models import (KINDS, ROOT, TransformerPolicy, build, default_checkpoint,
                                   detach_memory, load, save)

torch.set_num_threads(1)


class BaselineTests(unittest.TestCase):
    def test_bundled_weights_are_untrained_and_match_fresh_seed42(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                brain, checkpoint = load(default_checkpoint(kind), kind)
                self.assertEqual(checkpoint["training"]["status"], "untrained")
                self.assertEqual(checkpoint["training"]["episodes"], 0)
                self.assertEqual(checkpoint["training"]["optimizer_updates"], 0)
                torch.manual_seed(42)
                fresh = build(kind)
                for key, value in fresh.state_dict().items():
                    # QR used by orthogonal actor initialization can differ at the
                    # last few bits between Windows and Apple's numerical libraries.
                    self.assertTrue(torch.allclose(value, brain.state_dict()[key], atol=1e-6, rtol=1e-5), key)

    def test_three_distinct_backbones_and_memory_reset(self):
        expected = (torch.nn.GRUCell, torch.nn.LSTMCell, torch.nn.TransformerEncoder)
        for kind, module in zip(KINDS, expected):
            with self.subTest(kind=kind):
                brain = build(kind).eval()
                self.assertIsInstance(brain.memory, module)
                observation = ContinuousMaze().observe()
                with torch.inference_mode():
                    first = brain(observation, brain.initial_memory())[0]
                    memory = brain(observation, brain.initial_memory())[2]
                    for _ in range(3):
                        _, _, memory = brain(observation, memory)
                    reset = brain(observation, brain.initial_memory())[0]
                self.assertTrue(torch.equal(first, reset))
                if kind == "lstm":
                    self.assertEqual(len(memory), 2)
                    self.assertEqual(memory[1].shape, (1, 64))

    def test_real_environment_feedback_and_action_bounds(self):
        for kind in KINDS:
            brain, _ = load(default_checkpoint(kind), kind)
            env = ContinuousMaze(max_steps=12)
            observation, memory = env.reset(), brain.initial_memory()
            before = {k: v.clone() for k, v in brain.state_dict().items()}
            with torch.inference_mode():
                for _ in range(12):
                    action, _, _, _, memory = brain.act(observation, memory, greedy=True)
                    self.assertTrue(0 <= action[0] <= 1 and -1 <= action[1] <= 1)
                    observation, _, terminated, truncated, _ = env.step(action)
                    self.assertEqual(len(observation), 31)
                    self.assertEqual(observation[-3:-1], action)
                    if terminated or truncated:
                        break
            self.assertGreater(env.path_length, 0)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, brain.state_dict()[key]))

    def test_transformer_context_is_bounded_and_future_is_masked(self):
        brain = TransformerPolicy(context=4).eval()
        history = torch.randn(1, 4, 31)
        altered = history.clone()
        altered[:, 2:] += 100
        with torch.inference_mode():
            original_output = brain.encode_history(history)
            altered_output = brain.encode_history(altered)
        self.assertTrue(torch.allclose(original_output[:, :2], altered_output[:, :2], atol=1e-6))
        memory = brain.initial_memory()
        observations = torch.randn(10, 31)
        for observation in observations:
            _, _, memory = brain(observation, memory)
        self.assertEqual(memory.shape, (1, 4, 31))
        self.assertTrue(torch.equal(memory[0], observations[-4:]))

    def test_lstm_detaches_both_states(self):
        brain = build("lstm")
        _, _, memory = brain(ContinuousMaze().observe(), brain.initial_memory())
        self.assertTrue(all(item.grad_fn is not None for item in memory))
        self.assertTrue(all(item.grad_fn is None for item in detach_memory(memory)))

    def test_checkpoint_roundtrip_and_wrong_architecture_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            brain = build("lstm").eval()
            env = ContinuousMaze()
            save(path, brain, env, dict(status="untrained", episodes=0, optimizer_updates=0))
            restored, _ = load(path, "lstm")
            self.assertTrue(torch.equal(brain(env.observe(), brain.initial_memory())[0],
                                        restored(env.observe(), restored.initial_memory())[0]))
            with self.assertRaises(ValueError):
                load(path, "gru")
            with self.assertRaises(FileExistsError):
                save(path, brain, env, {})

    def test_training_multi_updates_changes_each_backbone_but_not_bundles(self):
        hashes = {kind: hashlib.sha256(default_checkpoint(kind).read_bytes()).hexdigest() for kind in KINDS}
        with tempfile.TemporaryDirectory() as directory:
            for kind in KINDS:
                with self.subTest(kind=kind):
                    torch.manual_seed(91)
                    brain = build(kind, **({"context": 4} if kind == "transformer" else {}))
                    before = {k: v.clone() for k, v in brain.state_dict().items()}
                    path = Path(directory) / f"{kind}.pt"
                    result = train(ContinuousMaze(max_steps=7, seed=91), brain, path,
                                   episodes=2, seed=91, rollout=3)
                    self.assertEqual(result["optimizer_updates"], 6)
                    self.assertTrue(any(not torch.equal(before['memory.'+k], value)
                                        for k, value in brain.memory.state_dict().items()))
                    restored, checkpoint = load(path, kind)
                    self.assertEqual(checkpoint["training"]["status"], "trained")
                    self.assertEqual(evaluate(ContinuousMaze(max_steps=3), restored, episodes=1)["episodes"], 1)
        self.assertEqual(hashes, {kind: hashlib.sha256(default_checkpoint(kind).read_bytes()).hexdigest()
                                 for kind in KINDS})

    def test_diagonal_motion_collision_and_bad_map(self):
        env = ContinuousMaze(["#######", "#....G#", "#.....#", "#S....#", "#######"])
        env.fly.heading = -0.7853981633974483
        x, y = env.fly.x, env.fly.y
        env.step([1, 0])
        self.assertGreater(env.fly.x, x)
        self.assertLess(env.fly.y, y)
        for _ in range(80):
            if env.terminated or env.truncated:
                break
            env.step([1, 0])
        self.assertGreater(env.collisions, 0)
        with self.assertRaises(ValueError):
            ContinuousMaze(["#####", "#S#G#", "#####"])

    def test_all_cli_launchers_work_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for kind in KINDS:
                result = subprocess.run([sys.executable, str(ROOT / f"run_{kind}.py"), "check"],
                                        cwd=directory, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("CHECK OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
