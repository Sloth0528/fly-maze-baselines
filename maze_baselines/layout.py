"""Validate maps without exposing a route to the policy."""

from collections import deque

DEFAULT_MAP = ["#######", "###G###", "###.###", "#S....#", "#######"]


class Maze:
    def __init__(self, layout):
        self.layout = list(layout)
        if not self.layout or not self.layout[0]:
            raise ValueError("map is empty")
        self.height, self.width = len(self.layout), len(self.layout[0])
        if any(len(row) != self.width for row in self.layout):
            raise ValueError("map rows must have equal widths")
        if any(c not in "#.SG" for row in self.layout for c in row):
            raise ValueError("map accepts only # . S G")
        positions = lambda symbol: [(x, y) for y, row in enumerate(self.layout)
                                    for x, c in enumerate(row) if c == symbol]
        starts, goals = positions("S"), positions("G")
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("map needs exactly one S and one G")
        self.start, self.goal = starts[0], goals[0]
        seen, queue = {self.goal}, deque([self.goal])
        self.start_candidates = []
        while queue:
            x, y = queue.popleft()
            self.start_candidates.append((x, y))
            for nx, ny in ((x+1, y), (x, y+1), (x-1, y), (x, y-1)):
                if (0 <= nx < self.width and 0 <= ny < self.height
                        and self.layout[ny][nx] != "#" and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        if self.start not in seen:
            raise ValueError("S cannot reach G")
