"""Architecture-specific launchers share all environment and training defaults."""

import argparse
import json
from pathlib import Path
import struct
import platform

import torch

from .environment import ContinuousMaze, from_config
from .experiment import evaluate, train, write_json
from .models import KINDS, build, default_checkpoint, load, parameter_count, save


def platform_check():
    if struct.calcsize("P") != 8:
        raise ValueError("64-bit Python is required")
    if platform.system() == "Darwin" and platform.machine().lower() != "arm64":
        raise ValueError("This dependency profile requires native Apple Silicon arm64 Python on macOS")


def environment(args, config=None):
    if config is not None:
        return from_config(config, seed=args.seed, path=args.maze, difficulty=args.difficulty,
                           vary_maze=args.vary_maze, max_steps=args.max_steps)
    layout = Path(args.maze).read_text(encoding="utf-8-sig").splitlines() if args.maze else None
    return ContinuousMaze(layout, difficulty=args.difficulty, vary_maze=args.vary_maze,
                          seed=args.seed, max_steps=args.max_steps if args.max_steps is not None else 800)


def main(default_kind):
    parser = argparse.ArgumentParser(description=f"{default_kind.upper()} continuous maze baseline")
    parser.add_argument("command", nargs="?", default="preview",
                        choices=("preview", "check", "init", "train", "evaluate"))
    parser.add_argument("--checkpoint", help="preview/check/evaluate a saved model of this architecture")
    parser.add_argument("--output", help="unused .pt for init/train, unused .json for evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--maze", help="TXT map containing # . S G")
    maps.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    parser.add_argument("--vary-maze", action="store_true")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--sample", action="store_true", help="sample policy instead of greedy preview/evaluation")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--rollout", type=int, default=32)
    args = parser.parse_args()
    try:
        platform_check()
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        if args.command in ("init", "train"):
            if args.checkpoint:
                raise ValueError("init/train start from fresh weights; --checkpoint is for preview/check/evaluate")
            if not args.output:
                raise ValueError("init/train require --output with an unused .pt path")
            if Path(args.output).suffix != ".pt":
                raise ValueError("init/train output must end in .pt")
            brain, env = build(default_kind), environment(args)
            if args.command == "init":
                save(args.output, brain, env, dict(status="untrained", episodes=0,
                     optimizer_updates=0, seed=args.seed, algorithm="random_initialization",
                     trainable_parameters=parameter_count(brain)))
                print(f"Saved UNTRAINED {default_kind}: {args.output}")
            else:
                print("Training starts from fresh random weights; bundled models stay unchanged.")
                train(env, brain, args.output, episodes=args.episodes if args.episodes is not None else 1000,
                      seed=args.seed, lr=args.lr, rollout=args.rollout, random_start=args.random_start)
            return
        path = Path(args.checkpoint) if args.checkpoint else default_checkpoint(default_kind)
        brain, checkpoint = load(path, expected_kind=default_kind)
        env = environment(args, checkpoint["environment"])
        status = dict(architecture=brain.kind, checkpoint=str(path.resolve()),
                      parameters=parameter_count(brain), training=checkpoint["training"])
        print(json.dumps(status, indent=2))
        if args.command == "check":
            with torch.inference_mode():
                action, _, _, _, _ = brain.act(env.observe(), brain.initial_memory(), greedy=True)
                env.step(action)
            print("CHECK OK: finite action, environment step, checkpoint loaded")
        elif args.command == "evaluate":
            if args.output and Path(args.output).exists():
                raise FileExistsError(args.output)
            report = {**status, "test_seed": args.seed, "sample": args.sample,
                      "random_start": args.random_start, "environment": env.config(),
                      **evaluate(env, brain, episodes=args.episodes if args.episodes is not None else 20,
                                 random_start=args.random_start, sample=args.sample)}
            if args.output:
                write_json(args.output, report)
            print(f"Success: {report['successes']}/{report['episodes']}")
        else:
            from .viewer import Viewer
            Viewer(env, brain, checkpoint, path, sample=args.sample, random_start=args.random_start).run()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
