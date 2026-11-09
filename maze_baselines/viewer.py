"""Visible model/weight status; the policy alone supplies both motor actions."""

import math
import tkinter as tk
from tkinter import ttk

import torch


class Viewer:
    def __init__(self, env, brain, checkpoint, path, *, sample=False, random_start=False):
        self.env, self.brain = env, brain
        self.sample, self.random_start = sample, random_start
        self.root = tk.Tk()
        self.root.title(f"Maze Baselines - {brain.kind.upper()}")
        self.running = False
        self.cell = min(65, max(10, 700//max(env.width, env.height)))
        self.canvas = tk.Canvas(self.root, width=env.width*self.cell,
                                height=env.height*self.cell, bg="#f5f6f8", highlightthickness=0)
        self.canvas.pack(padx=15, pady=15)
        info = checkpoint["training"]
        self.label = (f"{brain.kind.upper()} | {info['status'].upper()} | "
                      f"episodes={info['episodes']} | updates={info['optimizer_updates']}\n{path.name}")
        ttk.Label(self.root, text=self.label, justify="center").pack(padx=15)
        self.status = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status, justify="center").pack(padx=15, pady=10)
        row = ttk.Frame(self.root)
        row.pack(pady=(0, 15))
        ttk.Button(row, text="Play / Pause", command=self.toggle).pack(side="left", padx=4)
        ttk.Button(row, text="One step", command=self.step).pack(side="left", padx=4)
        ttk.Button(row, text="Reset memory + maze", command=self.reset).pack(side="left", padx=4)
        ttk.Label(self.root, text="UNTRAINED weights may hit walls, circle, or stop. "
                  "Preview never trains.").pack(padx=15, pady=(0, 15))
        self.reset()
        self.root.after(100, self.tick)

    def reset(self):
        self.observation = self.env.reset(self.random_start)
        self.memory = self.brain.initial_memory()
        self.action = [0.0, 0.0]
        self.draw()

    def toggle(self):
        self.running = not self.running

    def step(self):
        if self.env.terminated or self.env.truncated:
            self.running = False
            return
        with torch.inference_mode():
            self.action, _, _, _, self.memory = self.brain.act(
                self.observation, self.memory, greedy=not self.sample)
            self.observation, _, _, _, _ = self.env.step(self.action)
        self.draw()

    def tick(self):
        if self.running:
            self.step()
        self.root.after(100, self.tick)

    def draw(self):
        c, env, canvas = self.cell, self.env, self.canvas
        canvas.delete("all")
        for y, row in enumerate(env.layout):
            for x, tile in enumerate(row):
                canvas.create_rectangle(x*c, y*c, (x+1)*c, (y+1)*c,
                                        fill="#263343" if tile == "#" else "#f5f6f8", outline="")
                if tile in "SG":
                    canvas.create_text((x+.5)*c, (y+.5)*c, text=tile,
                                       fill="#188454" if tile == "G" else "#68798c", font=("Arial", 16))
        if len(env.path) > 1:
            canvas.create_line(*[v*c for point in env.path for v in point], fill="#e8a536", width=2)
        x, y, heading = env.fly.x, env.fly.y, env.fly.heading
        for angle, distance, _ in env.sense():
            canvas.create_line(x*c, y*c, (x+distance*math.cos(angle))*c,
                               (y+distance*math.sin(angle))*c, fill="#bdcedd")
        r = env.radius
        canvas.create_oval((x-r)*c, (y-r)*c, (x+r)*c, (y+r)*c, fill="#287ed1", outline="")
        canvas.create_line(x*c, y*c, (x+.35*math.cos(heading))*c,
                           (y+.35*math.sin(heading))*c, fill="#0d2641", width=3)
        result = "GOAL" if env.terminated else ("TIME LIMIT" if env.truncated else "RUNNING")
        self.status.set(f"{result} | step={env.steps}/{env.max_steps} | collisions={env.collisions}\n"
                        f"throttle={self.action[0]:.3f} | steering={self.action[1]:.3f} | "
                        f"path={env.path_length:.2f} cells")

    def run(self):
        self.root.mainloop()
