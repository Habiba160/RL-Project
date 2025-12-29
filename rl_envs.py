import numpy as np


# Actions for grid environments: 0=up, 1=right, 2=down, 3=left
ACTION_NAMES = ["↑", "→", "↓", "←"]


class GridWorldEnv:
    """Simple deterministic GridWorld.

    - Agent moves on an n_rows x n_cols grid.
    - Top-left cell (0) is the start, bottom-right cell is the goal.
    - Reward = -1 per step until reaching the goal (reward 0).
    """

    def __init__(self, n_rows=4, n_cols=4):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_states = n_rows * n_cols
        self.n_actions = 4
        self.start_state = 0
        self.terminal_state = self.n_states - 1
        self.P = self._build_transition_model()
        self.state = self.start_state

    def _to_pos(self, s):
        return divmod(s, self.n_cols)

    def _to_state(self, row, col):
        return row * self.n_cols + col

    def _build_transition_model(self):
        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if s == self.terminal_state:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                row, col = self._to_pos(s)
                if a == 0:
                    row2, col2 = max(row - 1, 0), col
                elif a == 1:
                    row2, col2 = row, min(col + 1, self.n_cols - 1)
                elif a == 2:
                    row2, col2 = min(row + 1, self.n_rows - 1), col
                else:
                    row2, col2 = row, max(col - 1, 0)
                s2 = self._to_state(row2, col2)
                reward = 0.0 if s2 == self.terminal_state else -1.0
                done = s2 == self.terminal_state
                P[s][a] = [(1.0, s2, reward, done)]
        return P

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        prob, s2, r, done = self.P[self.state][action][0]
        self.state = s2
        return s2, r, done, {}


class FrozenLakeEnv:
    """Deterministic 4x4 FrozenLake-style environment.

    Tiles: S (start), F (frozen), H (hole), G (goal).
    Rewards: -1 at holes, +1 at goal, 0 otherwise.
    """

    def __init__(self):
        self.desc = [
            list("SFFF"),
            list("FHFH"),
            list("FFFH"),
            list("HFFG"),
        ]
        self.n_rows = len(self.desc)
        self.n_cols = len(self.desc[0])
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        self.start_state = 0
        self.goal_state = self.n_states - 1
        self.holes = {
            self._to_state(r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.desc[r][c] == "H"
        }
        self.P = self._build_transition_model()
        self.state = self.start_state

    def _to_pos(self, s):
        return divmod(s, self.n_cols)

    def _to_state(self, row, col):
        return row * self.n_cols + col

    def _build_transition_model(self):
        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                row, col = self._to_pos(s)
                if s in self.holes or s == self.goal_state:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                if a == 0:
                    row2, col2 = max(row - 1, 0), col
                elif a == 1:
                    row2, col2 = row, min(col + 1, self.n_cols - 1)
                elif a == 2:
                    row2, col2 = min(row + 1, self.n_rows - 1), col
                else:
                    row2, col2 = row, max(col - 1, 0)
                s2 = self._to_state(row2, col2)
                tile2 = self.desc[row2][col2]
                if tile2 == "H":
                    reward, done = -1.0, True
                elif tile2 == "G":
                    reward, done = 1.0, True
                else:
                    reward, done = 0.0, False
                P[s][a] = [(1.0, s2, reward, done)]
        return P

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        prob, s2, r, done = self.P[self.state][action][0]
        self.state = s2
        return s2, r, done, {}


class BreakoutEnv:
    """Simple 1D Breakout-like environment.

    States are positions on a line; 0 is a hole (lose), last index is goal.
    Actions: 0=left, 1=stay, 2=right.
    """

    def __init__(self, length=7):
        self.length = length
        self.n_states = length
        self.n_actions = 3
        self.start_state = length // 2
        self.hole_state = 0
        self.goal_state = length - 1
        self.P = self._build_transition_model()
        self.state = self.start_state

    def _build_transition_model(self):
        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if s == self.hole_state or s == self.goal_state:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                if a == 0:
                    s2 = max(s - 1, 0)
                elif a == 2:
                    s2 = min(s + 1, self.length - 1)
                else:
                    s2 = s
                if s2 == self.goal_state:
                    reward, done = 1.0, True
                elif s2 == self.hole_state:
                    reward, done = -1.0, True
                else:
                    reward, done = -0.1, False
                P[s][a] = [(1.0, s2, reward, done)]
        return P

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        prob, s2, r, done = self.P[self.state][action][0]
        self.state = s2
        return s2, r, done, {}


class Gym4ReaLEnv:
    """Simple 1D custom line environment.

    Similar to BreakoutEnv but with no step cost (only terminal rewards).
    """

    def __init__(self, length=5):
        self.length = length
        self.n_states = length
        self.n_actions = 3
        self.start_state = length // 2
        self.hole_state = 0
        self.goal_state = length - 1
        self.P = self._build_transition_model()
        self.state = self.start_state

    def _build_transition_model(self):
        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if s == self.hole_state or s == self.goal_state:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                if a == 0:
                    s2 = max(s - 1, 0)
                elif a == 2:
                    s2 = min(s + 1, self.length - 1)
                else:
                    s2 = s
                if s2 == self.goal_state:
                    reward, done = 1.0, True
                elif s2 == self.hole_state:
                    reward, done = -1.0, True
                else:
                    reward, done = 0.0, False
                P[s][a] = [(1.0, s2, reward, done)]
        return P

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        prob, s2, r, done = self.P[self.state][action][0]
        self.state = s2
        return s2, r, done, {}


def make_environment(name: str):
    """Factory that returns an environment instance by human-readable name."""
    if name == "GridWorld 4x4":
        return GridWorldEnv(4, 4)
    if name == "FrozenLake 4x4":
        return FrozenLakeEnv()
    if name == "Breakout (custom line)":
        return BreakoutEnv(length=7)
    if name == "Gym4ReaL (custom line)":
        return Gym4ReaLEnv(length=5)
    raise ValueError(f"Unknown environment: {name}")
