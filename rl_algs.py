import numpy as np


# ========= Dynamic Programming algorithms =========


def policy_evaluation(env, policy, gamma=0.99, theta=1e-4, max_iters=1000):
    """Iterative policy evaluation for a given policy and model env.P."""
    V = np.zeros(env.n_states)
    for _ in range(max_iters):
        delta = 0.0
        for s in range(env.n_states):
            v = 0.0
            for a, pi_sa in enumerate(policy[s]):
                for prob, s2, r, done in env.P[s][a]:
                    v += pi_sa * prob * (r + gamma * (0 if done else V[s2]))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def greedy_policy_improvement(env, V, gamma=0.99):
    """Greedy policy improvement given a value function and model env.P."""
    policy = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        q = np.zeros(env.n_actions)
        for a in range(env.n_actions):
            for prob, s2, r, done in env.P[s][a]:
                q[a] += prob * (r + gamma * (0 if done else V[s2]))
        best_a = int(np.argmax(q))
        policy[s, best_a] = 1.0
    return policy


def policy_iteration(env, gamma=0.99, theta=1e-4, max_iters=50):
    """Classic policy iteration (evaluate + improve until stable)."""
    policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
    for _ in range(max_iters):
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        new_policy = greedy_policy_improvement(env, V, gamma=gamma)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, V


def value_iteration(env, gamma=0.99, theta=1e-4, max_iters=1000):
    """Value iteration; returns optimal policy and value function."""
    V = np.zeros(env.n_states)
    for _ in range(max_iters):
        delta = 0.0
        for s in range(env.n_states):
            q = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                for prob, s2, r, done in env.P[s][a]:
                    q[a] += prob * (r + gamma * (0 if done else V[s2]))
            v_new = np.max(q)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    policy = greedy_policy_improvement(env, V, gamma=gamma)
    return policy, V


# ========= Model-free algorithms =========


def epsilon_greedy(Q, state, epsilon):
    """Epsilon-greedy action selection from Q-table."""
    nA = Q.shape[1]
    if np.random.rand() < epsilon:
        return np.random.randint(nA)
    return int(np.argmax(Q[state]))


def monte_carlo_control(env, episodes=300, gamma=0.99, epsilon=0.1, max_steps=100):
    """Every-visit Monte Carlo control with epsilon-greedy exploration."""
    nS, nA = env.n_states, env.n_actions
    Q = np.zeros((nS, nA))
    returns_sum = np.zeros((nS, nA))
    returns_count = np.zeros((nS, nA))
    episode_returns = []

    for _ in range(episodes):
        episode = []
        s = env.reset()
        for t in range(max_steps):
            a = epsilon_greedy(Q, s, epsilon)
            s2, r, done, _ = env.step(a)
            episode.append((s, a, r))
            s = s2
            if done:
                break
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[t]
            G = gamma * G + r_t
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                returns_sum[s_t, a_t] += G
                returns_count[s_t, a_t] += 1.0
                Q[s_t, a_t] = returns_sum[s_t, a_t] / max(1.0, returns_count[s_t, a_t])
        episode_returns.append(G)

    policy = np.zeros_like(Q)
    best_actions = np.argmax(Q, axis=1)
    policy[np.arange(nS), best_actions] = 1.0
    return policy, Q, episode_returns


def td0_prediction(env, policy, episodes=300, alpha=0.1, gamma=0.99, max_steps=100):
    """TD(0) prediction for a fixed policy."""
    V = np.zeros(env.n_states)
    episode_returns = []
    for _ in range(episodes):
        s = env.reset()
        G = 0.0
        for t in range(max_steps):
            a = np.random.choice(env.n_actions, p=policy[s])
            s2, r, done, _ = env.step(a)
            G += (gamma ** t) * r
            target = r + gamma * (0 if done else V[s2])
            V[s] += alpha * (target - V[s])
            s = s2
            if done:
                break
        episode_returns.append(G)
    return V, episode_returns


def n_step_td_prediction(env, policy, n=3, episodes=200, alpha=0.1, gamma=0.99, max_steps=100):
    """n-step TD prediction with finite horizon to avoid overflow."""
    V = np.zeros(env.n_states)
    episode_returns = []
    for _ in range(episodes):
        s0 = env.reset()
        states = [s0]
        rewards = [0.0]
        T = max_steps
        t = 0
        done = False
        while True:
            if t < max_steps and not done:
                s_t = states[-1]
                a_t = np.random.choice(env.n_actions, p=policy[s_t])
                s_next, r_next, done, _ = env.step(a_t)
                states.append(s_next)
                rewards.append(r_next)
                if done:
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                G = 0.0
                upper = min(tau + n, T)
                for i in range(tau + 1, upper + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]
                if tau + n < T:
                    G += (gamma ** n) * V[states[tau + n]]
                s_tau = states[tau]
                V[s_tau] += alpha * (G - V[s_tau])
            if tau >= T - 1:
                break
            t += 1
        G_ep = 0.0
        for i, r in enumerate(rewards[1:]):
            G_ep += (gamma ** i) * r
        episode_returns.append(G_ep)
    return V, episode_returns


def sarsa(env, episodes=300, alpha=0.1, gamma=0.99, epsilon=0.1, max_steps=100):
    """On-policy SARSA control."""
    nS, nA = env.n_states, env.n_actions
    Q = np.zeros((nS, nA))
    episode_returns = []
    for _ in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon)
        G = 0.0
        for t in range(max_steps):
            s2, r, done, _ = env.step(a)
            G += (gamma ** t) * r
            if done:
                td_target = r
                Q[s, a] += alpha * (td_target - Q[s, a])
                break
            a2 = epsilon_greedy(Q, s2, epsilon)
            td_target = r + gamma * Q[s2, a2]
            Q[s, a] += alpha * (td_target - Q[s, a])
            s, a = s2, a2
        episode_returns.append(G)
    policy = np.zeros_like(Q)
    best_actions = np.argmax(Q, axis=1)
    policy[np.arange(nS), best_actions] = 1.0
    return policy, Q, episode_returns


def q_learning(env, episodes=300, alpha=0.1, gamma=0.99, epsilon=0.1, max_steps=100):
    """Off-policy Q-learning control."""
    nS, nA = env.n_states, env.n_actions
    Q = np.zeros((nS, nA))
    episode_returns = []
    for _ in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon)
        G = 0.0
        for t in range(max_steps):
            s2, r, done, _ = env.step(a)
            G += (gamma ** t) * r
            best_next = 0.0 if done else np.max(Q[s2])
            td_target = r + gamma * best_next
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s2
            if done:
                break
            a = epsilon_greedy(Q, s, epsilon)
        episode_returns.append(G)
    policy = np.zeros_like(Q)
    best_actions = np.argmax(Q, axis=1)
    policy[np.arange(nS), best_actions] = 1.0
    return policy, Q, episode_returns
