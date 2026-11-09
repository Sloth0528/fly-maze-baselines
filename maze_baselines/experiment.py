"""Shared CPU Actor-Critic training and read-only evaluation."""

import csv
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from .models import detach_memory, parameter_count, save


def train(env, brain, output, *, episodes=1000, seed=42, lr=0.0003, rollout=32,
          gamma=0.99, random_start=False):
    output = Path(output)
    log = output.with_suffix(".csv")
    if output.suffix != ".pt" or output.exists() or log.exists():
        raise ValueError("choose an unused .pt output and companion .csv")
    if episodes < 1 or rollout < 1 or not math.isfinite(lr) or lr <= 0 or not 0 <= gamma <= 1:
        raise ValueError("invalid training settings")
    output.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(brain.parameters(), lr=lr)
    brain.train()
    started, updates = time.perf_counter(), 0
    fields = ["episode", "success", "steps", "reward", "collisions", "path_length", "loss"]
    with log.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode_index in range(1, episodes+1):
            observation, memory = env.reset(random_start), brain.initial_memory()
            total_reward, losses = 0.0, []
            while not (env.terminated or env.truncated):
                rewards, log_probs, values, entropies = [], [], [], []
                for _ in range(rollout):
                    action, log_prob, value, entropy, memory = brain.act(observation, memory)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    values.append(value)
                    entropies.append(entropy)
                    total_reward += reward
                    if terminated or truncated:
                        break
                with torch.no_grad():
                    total = 0.0 if env.terminated else brain(observation, memory)[1].item()
                targets = []
                for reward in reversed(rewards):
                    total = reward+gamma*total
                    targets.append(total)
                advantage = torch.tensor(targets[::-1])-torch.stack(values)
                loss = (-(torch.stack(log_probs)*advantage.detach()).mean()
                        +0.5*advantage.square().mean()-0.01*torch.stack(entropies).mean())
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(brain.parameters(), 0.5, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                losses.append(loss.item())
                memory = detach_memory(memory)
            writer.writerow(dict(episode=episode_index, success=int(env.terminated), steps=env.steps,
                                 reward=total_reward, collisions=env.collisions, path_length=env.path_length,
                                 loss=sum(losses)/len(losses)))
            handle.flush()
            if episode_index == 1 or episode_index % 100 == 0 or episode_index == episodes:
                print(f"{brain.kind}: episode {episode_index}/{episodes}, success={env.terminated}, "
                      f"reward={total_reward:.3f}, updates={updates}", flush=True)
    training = dict(status="trained", episodes=episodes, optimizer_updates=updates,
                    seed=seed, algorithm="actor_critic_tbptt", rollout=rollout, lr=lr,
                    gamma=gamma, random_start=random_start, wall_seconds=time.perf_counter()-started,
                    trainable_parameters=parameter_count(brain), entropy="base Gaussian")
    save(output, brain, env, training)
    return training


def evaluate(env, brain, *, episodes=20, random_start=False, sample=False):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    brain.eval()
    results = []
    with torch.inference_mode():
        for _ in range(episodes):
            observation, memory = env.reset(random_start), brain.initial_memory()
            total = 0.0
            while not (env.terminated or env.truncated):
                action, _, _, _, memory = brain.act(observation, memory, greedy=not sample)
                observation, reward, _, _, _ = env.step(action)
                total += reward
            results.append(dict(success=env.terminated, steps=env.steps,
                                path_length=env.path_length, collisions=env.collisions,
                                reward=total, initial_pose=env.initial_pose, map_seed=env.map_seed))
    successes = sum(r["success"] for r in results)
    return dict(episodes=episodes, successes=successes, success_rate=successes/episodes,
                mean_steps=sum(r["steps"] for r in results)/episodes, results=results)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
