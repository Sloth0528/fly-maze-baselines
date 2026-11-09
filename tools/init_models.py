"""Generate reproducible random weights only; no optimizer or training import."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from maze_baselines.environment import ContinuousMaze
from maze_baselines.models import KINDS, build, parameter_count, save


def main():
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    destinations = [directory / f"{kind}_untrained.pt" for kind in KINDS]
    manifest_path = directory / "manifest.json"
    if any(path.exists() for path in [*destinations, manifest_path]):
        raise FileExistsError("model generation refuses to overwrite existing defaults")
    torch.set_num_threads(1)
    manifest = dict(project_version="1.0.0", torch_version=str(torch.__version__), seed=42, models={})
    for kind, path in zip(KINDS, destinations):
        torch.manual_seed(42)
        brain, env = build(kind), ContinuousMaze(seed=42)
        training = dict(status="untrained", episodes=0, optimizer_updates=0, seed=42,
                        algorithm="random_initialization", trainable_parameters=parameter_count(brain))
        save(path, brain, env, training)
        manifest["models"][kind] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      brain=brain.config(), **training)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
