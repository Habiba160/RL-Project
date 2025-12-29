import numpy as np
import streamlit as st

from rl_envs import make_environment
from rl_algs import (
    policy_evaluation,
    policy_iteration,
    value_iteration,
    monte_carlo_control,
    td0_prediction,
    n_step_td_prediction,
    sarsa,
    q_learning,
)
from rl_utils import (
    plot_grid_values,
    plot_episode_returns,
    run_greedy_episode,
    episode_trace_to_str,
)


st.set_page_config(page_title="Interactive RL Lab", layout="centered")
st.title("Interactive Reinforcement Learning (RL) Lab")
st.write(
    "This web tool lets you experiment with simple tabular RL environments "
    "and algorithms. Choose an environment and algorithm, set the parameters, "
    "and run training to visualize learning and agent behaviour."
)


# ----- Sidebar controls -----

st.sidebar.header("Configuration")

env_name = st.sidebar.selectbox(
    "Environment",
    [
        "GridWorld 4x4",
        "FrozenLake 4x4",
        "Breakout (custom line)",
        "Gym4ReaL (custom line)",
    ],
)

alg_name = st.sidebar.selectbox(
    "Algorithm",
    [
        "Policy Evaluation",
        "Policy Iteration",
        "Value Iteration",
        "Monte Carlo (MC)",
        "TD(0)",
        "n-step TD",
        "SARSA",
        "Q-learning",
    ],
)

gamma = st.sidebar.slider("Discount factor (gamma)", 0.0, 0.999, 0.99, 0.01)
alpha = st.sidebar.slider("Learning rate (alpha)", 0.001, 1.0, 0.1, 0.001)
epsilon = st.sidebar.slider("Exploration rate (epsilon)", 0.0, 1.0, 0.1, 0.05)

episodes = st.sidebar.slider("Episodes", 10, 2000, 200, 10)
max_steps = st.sidebar.slider("Max steps per episode", 10, 500, 50, 10)

n_step_n = st.sidebar.slider("n (for n-step TD)", 2, 10, 3, 1)

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1)


# ----- Main run button -----

run = st.button("Run training")


def set_seeds(seed_value: int):
    np.random.seed(seed_value)


if run:
    set_seeds(seed)

    st.subheader("Run summary")
    st.write(f"**Environment:** {env_name}")
    st.write(f"**Algorithm:** {alg_name}")
    st.write(
        f"**gamma** = {gamma:.3f}, **alpha** = {alpha:.3f}, "
        f"**epsilon** = {epsilon:.3f}, **episodes** = {episodes}, "
        f"**max steps** = {max_steps}"
    )

    # Create environment instance
    try:
        env = make_environment(env_name)
    except Exception as e:  # pragma: no cover - defensive
        st.error(f"Could not create environment: {e}")
        st.stop()

    # ----- Algorithm dispatch -----

    if alg_name == "Policy Evaluation":
        policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
        V = policy_evaluation(env, policy, gamma=gamma)
        st.markdown("### Value function under a uniform random policy")
        if hasattr(env, "n_rows") and hasattr(env, "n_cols"):
            fig = plot_grid_values(env, V, policy, title="Policy Evaluation: V and π")
            st.pyplot(fig)
        else:
            st.write(V)

    elif alg_name == "Policy Iteration":
        policy, V = policy_iteration(env, gamma=gamma)
        st.markdown("### Optimal value function and policy (Policy Iteration)")
        if hasattr(env, "n_rows") and hasattr(env, "n_cols"):
            fig = plot_grid_values(env, V, policy, title="Policy Iteration: optimal V and π")
            st.pyplot(fig)
        states, actions, rewards = run_greedy_episode(env, policy, max_steps=max_steps)
        trace = episode_trace_to_str(states, actions, rewards, title=env_name)
        st.markdown("### Greedy episode trace")
        st.code(trace)

    elif alg_name == "Value Iteration":
        policy, V = value_iteration(env, gamma=gamma)
        st.markdown("### Optimal value function and policy (Value Iteration)")
        if hasattr(env, "n_rows") and hasattr(env, "n_cols"):
            fig = plot_grid_values(env, V, policy, title="Value Iteration: optimal V and π")
            st.pyplot(fig)
        states, actions, rewards = run_greedy_episode(env, policy, max_steps=max_steps)
        trace = episode_trace_to_str(states, actions, rewards, title=env_name)
        st.markdown("### Greedy episode trace")
        st.code(trace)

    elif alg_name == "Monte Carlo (MC)":
        policy, Q, returns = monte_carlo_control(
            env,
            episodes=episodes,
            gamma=gamma,
            epsilon=epsilon,
            max_steps=max_steps,
        )
        st.markdown("### Monte Carlo control: learning curve")
        fig = plot_episode_returns(returns, title="MC control: returns")
        st.pyplot(fig)
        if hasattr(env, "n_rows") and hasattr(env, "n_cols"):
            V = np.max(Q, axis=1)
            fig_v = plot_grid_values(env, V, policy, title="MC: V and greedy π")
            st.pyplot(fig_v)
        states, actions, rewards = run_greedy_episode(env, policy, max_steps=max_steps)
        trace = episode_trace_to_str(states, actions, rewards, title=env_name)
        st.markdown("### Greedy episode trace")
        st.code(trace)

    elif alg_name == "TD(0)":
        policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
        V, returns = td0_prediction(
            env,
            policy,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            max_steps=max_steps,
        )
        st.markdown("### TD(0) prediction: learning curve")
        fig = plot_episode_returns(returns, title="TD(0) prediction: returns")
        st.pyplot(fig)
        if hasattr(env, "n_rows") and hasattr(env, "n_cols"):
            fig_v = plot_grid_values(env, V, None, title="TD(0): V under random policy")
            st.pyplot(fig_v)
        else:
            st.write(V)

    elif alg_name == "n-step TD":
        policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
        V, returns = n_step_td_prediction(
            env,
            policy,
            n=n_step_n,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            max_steps=max_steps,
        )
        st.markdown("### n-step TD prediction: learning curve")
        fig = plot_episode_returns(returns, title="n-step TD prediction: returns")
        st.pyplot(fig)
        if hasattr(env, "n_rows") and hasattr(env, "n_cols"):
            fig_v = plot_grid_values(env, V, None, title="n-step TD: V under random policy")
            st.pyplot(fig_v)
        else:
            st.write(V)

    elif alg_name == "SARSA":
        policy, Q, returns = sarsa(
            env,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            max_steps=max_steps,
        )
        st.markdown("### SARSA control: learning curve")
        fig = plot_episode_returns(returns, title="SARSA control: returns")
        st.pyplot(fig)
        if hasattr(env, "n_rows") and hasattr(env, "n_cols"):
            V = np.max(Q, axis=1)
            fig_v = plot_grid_values(env, V, policy, title="SARSA: V and greedy π")
            st.pyplot(fig_v)
        states, actions, rewards = run_greedy_episode(env, policy, max_steps=max_steps)
        trace = episode_trace_to_str(states, actions, rewards, title=env_name)
        st.markdown("### Greedy episode trace")
        st.code(trace)

    elif alg_name == "Q-learning":
        policy, Q, returns = q_learning(
            env,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            max_steps=max_steps,
        )
        st.markdown("### Q-learning control: learning curve")
        fig = plot_episode_returns(returns, title="Q-learning control: returns")
        st.pyplot(fig)
        if hasattr(env, "n_rows") and hasattr(env, "n_cols"):
            V = np.max(Q, axis=1)
            fig_v = plot_grid_values(env, V, policy, title="Q-learning: V and greedy π")
            st.pyplot(fig_v)
        states, actions, rewards = run_greedy_episode(env, policy, max_steps=max_steps)
        trace = episode_trace_to_str(states, actions, rewards, title=env_name)
        st.markdown("### Greedy episode trace")
        st.code(trace)

    else:  # pragma: no cover - defensive
        st.error("Unknown algorithm selected.")
