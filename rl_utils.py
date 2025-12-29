import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from rl_envs import ACTION_NAMES


def plot_grid_values(env, values, policy=None, title="Value Function"):
    """Create a heatmap figure for a grid environment's value function.

    If a policy is given, arrows are drawn indicating the greedy action.
    """
    grid = values.reshape(env.n_rows, env.n_cols)

    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = colors.LinearSegmentedColormap.from_list("vals", ["#f7fbff", "#08306b"])
    im = ax.imshow(grid, cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = r * env.n_cols + c
            ax.text(c, r, f"{grid[r, c]:.1f}", ha="center", va="center", color="black", fontsize=8)
            if policy is not None:
                a = int(np.argmax(policy[s]))
                ax.text(
                    c,
                    r + 0.25,
                    ACTION_NAMES[a],
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=9,
                )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_episode_returns(returns, title="Episode returns"):
    """Create a simple learning-curve figure (episode index vs return)."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(returns, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def run_greedy_episode(env, policy, max_steps=50):
    """Run one episode using a greedy policy w.r.t. a Q-table or tabular policy.

    Returns
    -------
    states : list[int]
    actions : list[int]
    rewards : list[float]
    """
    states, actions, rewards = [], [], []
    s = env.reset()
    for _ in range(max_steps):
        a = int(np.argmax(policy[s]))
        s2, r, done, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s2
        if done:
            states.append(s)
            break
    return states, actions, rewards


def episode_trace_to_str(states, actions, rewards, title="Episode"):
    """Format an episode (state, action, reward) as a multi-line string."""
    lines = [f"=== {title} ==="]
    for t in range(len(actions)):
        lines.append(f"t={t:02d}  s={states[t]}  a={actions[t]}  r={rewards[t]:.2f}")
    if states:
        lines.append(f"Terminal state: {states[-1]}")
    return "\n".join(lines)
