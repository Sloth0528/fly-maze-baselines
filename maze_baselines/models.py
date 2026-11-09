"""Three temporal backbones, one continuous actor-critic interface."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from . import __version__

KINDS = ("gru", "lstm", "transformer")
ROOT = Path(__file__).resolve().parents[1]


def detach_memory(memory):
    if isinstance(memory, tuple):
        return tuple(detach_memory(item) for item in memory)
    return memory.detach()


class Policy(nn.Module):
    def init_heads(self):
        self.actor = nn.Linear(self.hidden_size, 2)
        self.critic = nn.Linear(self.hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def tensor(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32,
                                      device=self.log_std.device)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[1] != self.observation_size:
            raise ValueError("expected [batch, 31] local observations")
        if not torch.isfinite(observation).all():
            raise ValueError("non-finite observation")
        return observation

    def heads(self, hidden, memory):
        return self.actor(hidden), self.critic(hidden).squeeze(-1), memory

    def act(self, observation, memory, greedy=False):
        mean, value, memory = self(observation, memory)
        if mean.shape[0] != 1:
            raise ValueError("act supports one environment at a time")
        distribution = Normal(mean, self.log_std.clamp(-3, 0.5).exp())
        z = mean if greedy else distribution.sample()
        bounded = z.tanh()
        log_prob = (distribution.log_prob(z)-torch.log(1-bounded.square()+1e-6)).sum()
        entropy = distribution.entropy().sum()  # base Gaussian entropy
        return ([(bounded[0, 0].item()+1)/2, bounded[0, 1].item()],
                log_prob, value.squeeze(), entropy, memory)

    def config(self):
        return {"kind": self.kind, "observation_size": self.observation_size,
                "hidden_size": self.hidden_size}


class GRUPolicy(Policy):
    kind = "gru"

    def __init__(self, observation_size=31, hidden_size=64):
        super().__init__()
        self.observation_size, self.hidden_size = observation_size, hidden_size
        self.encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        self.memory = nn.GRUCell(hidden_size, hidden_size)
        self.init_heads()

    def initial_memory(self, batch_size=1):
        return self.log_std.new_zeros(batch_size, self.hidden_size)

    def forward(self, observation, memory):
        hidden = self.memory(self.encoder(self.tensor(observation)), memory)
        return self.heads(hidden, hidden)


class LSTMPolicy(Policy):
    kind = "lstm"

    def __init__(self, observation_size=31, hidden_size=64):
        super().__init__()
        self.observation_size, self.hidden_size = observation_size, hidden_size
        self.encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        self.memory = nn.LSTMCell(hidden_size, hidden_size)
        self.init_heads()

    def initial_memory(self, batch_size=1):
        return tuple(self.log_std.new_zeros(batch_size, self.hidden_size) for _ in range(2))

    def forward(self, observation, memory):
        hidden, cell = self.memory(self.encoder(self.tensor(observation)), memory)
        return self.heads(hidden, (hidden, cell))


class TransformerPolicy(Policy):
    kind = "transformer"

    def __init__(self, observation_size=31, hidden_size=64, context=32, heads=4, layers=2):
        super().__init__()
        if context < 1 or heads < 1 or layers < 1 or hidden_size % heads:
            raise ValueError("invalid Transformer dimensions")
        self.observation_size, self.hidden_size = observation_size, hidden_size
        self.context, self.attention_heads, self.layers = context, heads, layers
        self.encoder = nn.Linear(observation_size, hidden_size)
        self.position = nn.Embedding(context, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, heads, hidden_size*2,
                                           dropout=0.0, batch_first=True)
        self.memory = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        # TransformerEncoder clones a layer; initialize independent layer weights.
        for block in self.memory.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.init_heads()

    def initial_memory(self, batch_size=1):
        return self.log_std.new_zeros(batch_size, 0, self.observation_size)

    def encode_history(self, history):
        length = history.shape[1]
        positions = torch.arange(length, device=history.device)
        tokens = self.encoder(history)+self.position(positions).unsqueeze(0)
        mask = torch.ones(length, length, dtype=torch.bool, device=history.device).triu(1)
        return self.memory(tokens, mask=mask)

    def forward(self, observation, memory):
        observation = self.tensor(observation)
        # Store raw observations, never a graph from a previous optimizer update.
        history = torch.cat((memory, observation.unsqueeze(1)), dim=1)[:, -self.context:]
        hidden = self.encode_history(history)[:, -1]
        return self.heads(hidden, history.detach())

    def config(self):
        return {**super().config(), "context": self.context,
                "heads": self.attention_heads, "layers": self.layers}


def build(kind, **config):
    return {"gru": GRUPolicy, "lstm": LSTMPolicy, "transformer": TransformerPolicy}[kind](**config)


def parameter_count(brain):
    return sum(p.numel() for p in brain.parameters() if p.requires_grad)


def save(path, brain, env, training):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save({"format": "maze-baselines-1", "project_version": __version__,
                    "brain": brain.config(), "environment": env.config(),
                    "training": training, "state_dict": brain.state_dict()}, handle)


def load(path, expected_kind=None):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "maze-baselines-1":
        raise ValueError("checkpoint belongs to a different project/format")
    config = dict(checkpoint["brain"])
    kind = config.pop("kind")
    if kind not in KINDS or (expected_kind and kind != expected_kind):
        raise ValueError("checkpoint architecture does not match launcher")
    if config["observation_size"] != 31:
        raise ValueError("unsupported observation interface")
    brain = build(kind, **config)
    brain.load_state_dict(checkpoint["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() for p in brain.parameters()):
        raise ValueError("non-finite checkpoint weights")
    brain.eval()
    return brain, checkpoint


def default_checkpoint(kind):
    directory = ROOT / "models"
    entry = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["models"][kind]
    path = (directory / entry["file"]).resolve()
    if path.parent != directory.resolve():
        raise ValueError("invalid default model path")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("default checkpoint checksum mismatch")
    return path
