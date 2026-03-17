"""
Simple Genetic Algorithm for highway-env
=========================================
Algorithm  : Fixed-topology MLP evolved via selection, crossover, mutation
Logging    : TensorBoard  →  ~/tb_logs/simple_ga/
Rendering  : Best agent shown every N generations
Platform   : Mac (sequential) / Linux (parallel via joblib)

Run:
    python ga_highway.py                         # defaults
    python ga_highway.py --pop 30 --gens 100     # faster
    tensorboard --logdir ~/tb_logs/simple_ga
"""

import os
import platform
import argparse
import time
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

# ══════════════════════════════════════════════════════════════════
# Environment config  (shared with DQN / PPO teammates)
# ══════════════════════════════════════════════════════════════════
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x":  [-100, 100],
            "y":  [-100, 100],
            "vx": [-20,   20],
            "vy": [-20,   20],
        },
        "absolute":  False,
        "order":     "sorted",
        "normalize": True,
    },
    "duration":             500,
    "simulation_frequency": 5,    # reduced from 15; 5 is enough for GA fitness
    "policy_frequency":     1,
    "collision_reward":            -10.0,
    "speed_reward_weight":          0.60,
    "lane_change_penalty":          0.12,
    "acceleration_penalty_weight":  0.04,
    "jerk_penalty_weight":          0.08,
    "action_acceleration_penalty":  0.05,
    "stable_speed_bonus":           0.03,
    "reward_speed_range": [20, 30],
}

SEED = 42
ACTION_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

# ══════════════════════════════════════════════════════════════════
# GA hyper-parameters
# ══════════════════════════════════════════════════════════════════
GA_CONFIG = {
    "population_size":  50,
    "n_generations":   100,
    "elite_frac":     0.20,
    "tournament_size":    5,
    "crossover_prob":  0.70,
    "mutation_std":    0.10,
    "mutation_decay":  0.995,
    "n_eval_episodes":    3,
    "render_every_n_gen": 10,
    "hidden_sizes":  [64, 64],
    "train_duration":   100,
    "eval_duration":    500,
}

# ══════════════════════════════════════════════════════════════════
# MLP Policy
# ══════════════════════════════════════════════════════════════════
class MLPPolicy:
    def __init__(self, obs_dim, n_actions, hidden_sizes):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_sizes = hidden_sizes
        sizes = [obs_dim] + hidden_sizes + [n_actions]
        self.shapes = [(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
        self.n_params = sum(w*h + h for w, h in self.shapes)

    def unpack(self, flat):
        layers, idx = [], 0
        for w_dim, h_dim in self.shapes:
            W = flat[idx: idx + w_dim*h_dim].reshape(w_dim, h_dim)
            idx += w_dim * h_dim
            b = flat[idx: idx + h_dim]; idx += h_dim
            layers.append((W, b))
        return layers

    def act(self, obs, flat):
        x = obs.flatten()
        layers = self.unpack(flat)
        for i, (W, b) in enumerate(layers):
            x = x @ W + b
            if i < len(layers) - 1:
                x = np.tanh(x)
        return int(np.argmax(x))


# ══════════════════════════════════════════════════════════════════
# Environment helpers
# ══════════════════════════════════════════════════════════════════
def make_env(render=False, duration=GA_CONFIG["train_duration"]):
    import highway_env as _h  # noqa — needed in subprocess workers
    env_id = "highway-v0" if render else "highway-fast-v0"
    env = gym.make(env_id, render_mode="human" if render else None)
    env.unwrapped.configure({**ENV_CONFIG, "duration": duration})
    env.reset(seed=SEED)
    return env


def evaluate_individual(flat, policy, n_episodes, render=False,
                        duration=GA_CONFIG["train_duration"]):
    env = make_env(render=render, duration=duration)
    ep_rewards, ep_lengths, ep_crashed = [], [], []
    all_speeds, all_lc, all_actions = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r, steps, crashed = 0.0, 0, False
        ep_speeds, lc, prev_lane = [], 0, None

        while True:
            a = policy.act(obs, flat)
            all_actions.append(a)
            obs, r, term, trunc, info = env.step(a)
            total_r += r; steps += 1
            ep_speeds.append(float(info.get("speed", 0.0)))
            lane = info.get("lane_index", None)
            if prev_lane is not None and lane != prev_lane:
                lc += 1
            prev_lane = lane
            if info.get("crashed", False):
                crashed = True
            if term or trunc:
                break

        ep_rewards.append(total_r); ep_lengths.append(steps)
        ep_crashed.append(float(crashed))
        all_speeds.extend(ep_speeds); all_lc.append(lc)

    env.close()
    speeds_arr  = np.array(all_speeds) if all_speeds else np.array([0.0])
    action_dist = np.bincount(all_actions, minlength=5) / max(len(all_actions), 1)

    return {
        "fitness":           float(np.mean(ep_rewards)),
        "mean_reward":       float(np.mean(ep_rewards)),
        "std_reward":        float(np.std(ep_rewards)),
        "mean_ep_length":    float(np.mean(ep_lengths)),
        "collision_rate":    float(np.mean(ep_crashed)),
        "mean_speed":        float(np.mean(speeds_arr)),
        "std_speed":         float(np.std(speeds_arr)),
        "mean_lane_changes": float(np.mean(all_lc)),
        "action_dist":       action_dist,
    }


# ══════════════════════════════════════════════════════════════════
# Population evaluation — multi-agent single environment
# All individuals drive simultaneously on the same road.
# NPCs simulated once → huge speedup vs N separate environments.
# ══════════════════════════════════════════════════════════════════
MA_ENV_CONFIG_EXTRA = {
    "action": {
        "type": "MultiAgentAction",
        "action_config": {"type": "DiscreteMetaAction"},
    },
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x":  [-100, 100], "y":  [-100, 100],
                "vx": [-20,   20], "vy": [-20,   20],
            },
            "absolute": False, "order": "sorted", "normalize": True,
        },
    },
}

def _per_vehicle_reward(env_u, action_i, vehicle):
    """Compute reward for a specific vehicle by temporarily swapping the pointer."""
    original    = env_u.vehicle
    env_u.vehicle = vehicle
    r = env_u._reward(action_i)
    env_u.vehicle = original
    return r

def evaluate_population(population, policy, n_episodes, duration, n_workers=1, render=False):
    """
    Evaluate all individuals in one multi-agent environment per episode.
    NPCs are simulated only once for the whole population → ~4x speedup.
    If render=True, opens a window showing all agents simultaneously.
    """
    import highway_env as _h  # noqa
    pop_size = len(population)

    # Per-individual accumulators  (across episodes)
    all_ep_rewards   = [[] for _ in range(pop_size)]
    all_ep_lengths   = [[] for _ in range(pop_size)]
    all_ep_crashed   = [[] for _ in range(pop_size)]
    all_speeds       = [[] for _ in range(pop_size)]
    all_lc           = [[] for _ in range(pop_size)]
    all_actions      = [[] for _ in range(pop_size)]

    ma_config = {
        **ENV_CONFIG,
        "duration":            duration,
        "controlled_vehicles": pop_size,
        "offroad_terminal":    False,   # don't freeze on crash
        **MA_ENV_CONFIG_EXTRA,
    }

    for ep in range(n_episodes):
        env_id = "highway-v0" if render else "highway-fast-v0"
        env = gym.make(env_id, render_mode="human" if render else None)
        env.unwrapped.configure(ma_config)
        obs_tuple, _ = env.reset(seed=SEED + ep)
        u = env.unwrapped
        u._is_terminated = lambda: False   # crashes don't end the episode

        ep_reward  = np.zeros(pop_size)
        ep_steps   = np.zeros(pop_size, dtype=int)
        ep_crashed = np.zeros(pop_size, dtype=bool)
        ep_speeds  = [[] for _ in range(pop_size)]
        ep_lc      = np.zeros(pop_size, dtype=int)
        prev_lanes = [None] * pop_size

        npcs = [v for v in u.road.vehicles if v not in set(u.controlled_vehicles)]

        while True:
            # Each individual picks its action from its own observation
            actions = tuple(
                policy.act(obs_tuple[i], population[i]) for i in range(pop_size)
            )
            for i, a in enumerate(actions):
                all_actions[i].append(a)

            obs_tuple, _, term, trunc, info = env.step(actions)

            # Unconditionally uncrash all NPCs every step —
            # NPCs should be indestructible; only agents can die.
            # (NPC-NPC crashes are also cleared, which is fine for GA training)
            for npc in npcs:
                npc.crashed = False

            # Compute per-vehicle rewards and metrics
            vehicles = u.controlled_vehicles
            for i in range(min(pop_size, len(vehicles))):
                v = vehicles[i]
                if not ep_crashed[i]:
                    r = _per_vehicle_reward(u, actions[i], v)
                    ep_reward[i] += r
                ep_steps[i]  += 1
                spd = float(v.speed)
                ep_speeds[i].append(spd)
                if v.crashed and not ep_crashed[i]:
                    ep_crashed[i] = True
                    _park_crashed_agent(v)
                lane = v.lane_index
                if prev_lanes[i] is not None and lane != prev_lanes[i]:
                    ep_lc[i] += 1
                prev_lanes[i] = lane

            # Only stop at duration end, not on crash
            if trunc:
                break

        env.close()

        for i in range(pop_size):
            all_ep_rewards[i].append(ep_reward[i])
            all_ep_lengths[i].append(int(ep_steps[i]))
            all_ep_crashed[i].append(float(ep_crashed[i]))
            all_speeds[i].extend(ep_speeds[i])
            all_lc[i].append(int(ep_lc[i]))

    # Build metrics dicts
    results = []
    for i in range(pop_size):
        acts       = all_actions[i]
        spd_arr    = np.array(all_speeds[i]) if all_speeds[i] else np.array([0.0])
        action_dist = np.bincount(acts, minlength=5) / max(len(acts), 1)
        results.append({
            "fitness":           float(np.mean(all_ep_rewards[i])),
            "mean_reward":       float(np.mean(all_ep_rewards[i])),
            "std_reward":        float(np.std(all_ep_rewards[i])),
            "mean_ep_length":    float(np.mean(all_ep_lengths[i])),
            "collision_rate":    float(np.mean(all_ep_crashed[i])),
            "mean_speed":        float(np.mean(spd_arr)),
            "std_speed":         float(np.std(spd_arr)),
            "mean_lane_changes": float(np.mean(all_lc[i])),
            "action_dist":       action_dist,
        })
    return results


# ══════════════════════════════════════════════════════════════════
# Genetic operators
# ══════════════════════════════════════════════════════════════════
def tournament_select(pop, fits, k):
    k = min(k, len(pop))
    idx = np.random.choice(len(pop), size=k, replace=False)
    return pop[idx[np.argmax(fits[idx])]].copy()

def crossover(a, b, prob):
    if np.random.rand() > prob: return a.copy()
    return np.where(np.random.rand(len(a)) < 0.5, a, b)

def mutate(ind, std):
    return ind + np.random.randn(*ind.shape) * std


# ══════════════════════════════════════════════════════════════════
# TensorBoard logging
# ══════════════════════════════════════════════════════════════════
def log_tb(writer, gen, pop_metrics, best_m, mutation_std, elapsed):
    fits        = [m["fitness"]        for m in pop_metrics]
    crash_rates = [m["collision_rate"] for m in pop_metrics]
    speeds      = [m["mean_speed"]     for m in pop_metrics]

    writer.add_scalar("fitness/best",            best_m["fitness"],    gen)
    writer.add_scalar("fitness/mean",            np.mean(fits),        gen)
    writer.add_scalar("fitness/std",             np.std(fits),         gen)
    writer.add_scalar("fitness/median",          np.median(fits),      gen)
    writer.add_scalar("fitness/worst",           np.min(fits),         gen)
    writer.add_histogram("fitness/distribution", np.array(fits),       gen)

    writer.add_scalar("reward/best_mean",        best_m["mean_reward"],gen)
    writer.add_scalar("reward/best_std",         best_m["std_reward"], gen)

    writer.add_scalar("behaviour/collision_rate",    best_m["collision_rate"],   gen)
    writer.add_scalar("behaviour/mean_speed",        best_m["mean_speed"],       gen)
    writer.add_scalar("behaviour/std_speed",         best_m["std_speed"],        gen)
    writer.add_scalar("behaviour/mean_lane_changes", best_m["mean_lane_changes"],gen)
    writer.add_scalar("behaviour/episode_length",    best_m["mean_ep_length"],   gen)

    writer.add_scalar("population/mean_crash_rate", np.mean(crash_rates), gen)
    writer.add_scalar("population/mean_speed",      np.mean(speeds),      gen)

    for i, name in enumerate(ACTION_NAMES):
        writer.add_scalar(f"action_dist/{name}", best_m["action_dist"][i], gen)

    writer.add_scalar("ga/mutation_std",    mutation_std, gen)
    writer.add_scalar("ga/seconds_per_gen", elapsed,      gen)
    writer.flush()


# ══════════════════════════════════════════════════════════════════
# Terminal display
# ══════════════════════════════════════════════════════════════════
C = {k: v for k, v in zip(
    ["cyan","green","yellow","red","bold","dim","reset"],
    ["\033[96m","\033[92m","\033[93m","\033[91m","\033[1m","\033[2m","\033[0m"]
)}

def sparkline(dist):
    blocks = " ▁▂▃▄▅▆▇█"
    parts  = []
    for v, name in zip(dist, ACTION_NAMES):
        idx = min(int(v * 40), 8)
        parts.append(f"{C['dim']}{name[:2]}{C['reset']}{blocks[idx]}")
    return " ".join(parts)

def print_header(cfg, policy, n_workers=1, is_linux=False):
    w = 72
    print(f"\n{C['bold']}{'═'*w}{C['reset']}")
    print(f"{C['bold']}  Simple Genetic Algorithm  ·  highway-env{C['reset']}")
    print(f"{'═'*w}")
    print(f"  Pop: {cfg['population_size']}  Gens: {cfg['n_generations']}  "
          f"Episodes: {cfg['n_eval_episodes']}  "
          f"Elite: {cfg['elite_frac']:.0%}  Tourn: {cfg['tournament_size']}")
    print(f"  Net: {policy.obs_dim}→{'→'.join(str(h) for h in cfg['hidden_sizes'])}→{policy.n_actions}"
          f"  ({policy.n_params:,} params)  "
          f"Train steps: {cfg['train_duration']}  Eval steps: {cfg['eval_duration']}")
    print(f"  Mutation σ: {cfg['mutation_std']} → ×{cfg['mutation_decay']}/gen  "
          f"Sim freq: {ENV_CONFIG['simulation_frequency']}  "
          f"Workers: {'joblib×' + str(n_workers) if is_linux else 'sequential (macOS)'}")
    print(f"{'─'*w}")
    print(f"  {'Gen':>7}  {'Best':>8}  {'Mean':>8}  {'Worst':>8}  "
          f"{'Crash':>6}  {'Speed':>6}  {'LC/ep':>5}  "
          f"{'σ':>6}  {'ETA':>7}  Actions")
    print(f"  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  "
          f"{'─'*6}  {'─'*6}  {'─'*5}  "
          f"{'─'*6}  {'─'*7}  {'─'*30}")

def print_gen(gen, n_gens, fits, best_m, mut_std, elapsed, eta, best_ever, improved):
    star      = f"{C['green']}★{C['reset']}" if improved else " "
    crash_c   = C['red'] if best_m["collision_rate"] > 0.5 else (
                C['yellow'] if best_m["collision_rate"] > 0 else C['green'])
    speed_c   = C['green'] if best_m["mean_speed"] >= 20 else C['yellow']
    eta_str   = f"{int(eta//60)}m{int(eta%60):02d}s" if eta > 0 else "  -"

    line = (
        f"{star} {gen+1:>5}/{n_gens}  "
        f"{C['bold']}{best_m['fitness']:>8.2f}{C['reset']}  "
        f"{np.mean(fits):>8.2f}  "
        f"{np.min(fits):>8.2f}  "
        f"{crash_c}{best_m['collision_rate']*100:>5.1f}%{C['reset']}  "
        f"{speed_c}{best_m['mean_speed']:>6.1f}{C['reset']}  "
        f"{best_m['mean_lane_changes']:>5.1f}  "
        f"{mut_std:>6.4f}  "
        f"{C['dim']}{eta_str:>7}{C['reset']}  "
        f"{sparkline(best_m['action_dist'])}"
    )
    print(line)

    # Extended summary every 10 gens
    if (gen + 1) % 10 == 0:
        print(
            f"  {C['dim']}  ↳ spread:{np.max(fits)-np.min(fits):>7.2f}  "
            f"median:{np.median(fits):>7.2f}  "
            f"std:{np.std(fits):>6.2f}  "
            f"best-ever:{best_ever:>7.2f}  "
            f"ep_len:{best_m['mean_ep_length']:>5.0f}  "
            f"spd_std:{best_m['std_speed']:>4.1f}  "
            f"time:{elapsed:.1f}s{C['reset']}"
        )




def _park_crashed_agent(v):
    """
    Cleanly remove a crashed agent from the simulation:
    - Reset all physics state to avoid residual forces
    - Teleport far off-road so it can't affect traffic
    """
    v.position              = np.array([-9999.0, -9999.0])
    v.speed                 = 0.0
    v.heading               = 0.0
    v.impact                = None                         # clear collision impulse
    v.action                = {"steering": 0.0, "acceleration": 0.0}
    v.crashed               = False                        # stop clip_actions braking



def _patch_agent_collisions(env_u):
    """
    Two patches to make agents ghosts to each other but real to NPCs:
    1. handle_collisions: skip agent-agent collision physics
    2. close_objects_to: exclude other agents from each agent's observation
    """
    import types
    controlled_set = set(env_u.controlled_vehicles)

    # --- Patch 1: skip agent-agent collision detection ---
    _orig_hc = env_u.controlled_vehicles[0].__class__.handle_collisions

    def _handle_collisions_filtered(self, other, dt=0):
        if other in controlled_set:
            return   # agents are ghosts to each other
        return _orig_hc(self, other, dt)

    for v in env_u.controlled_vehicles:
        v.handle_collisions = types.MethodType(_handle_collisions_filtered, v)

    # --- Patch 2: exclude other agents from observations ---
    _orig_close = env_u.road.__class__.close_objects_to

    def _close_filtered(self, vehicle, distance, **kwargs):
        result = _orig_close(self, vehicle, distance, **kwargs)
        return [v for v in result if v not in controlled_set]

    env_u.road.close_objects_to = types.MethodType(_close_filtered, env_u.road)

def _align_agents_same_start(env_u, start_x=100.0):
    """
    Place ALL controlled agents at exactly the same position.
    Collision between agents is handled by _patch_agent_collisions.
    """
    for v in env_u.controlled_vehicles:
        v.position = np.array([start_x, 0.0])
        v.speed    = 25.0
        v.heading  = 0.0
        v.crashed  = False

# ══════════════════════════════════════════════════════════════════
# Debug render: show all agents live in a single window
# ══════════════════════════════════════════════════════════════════
def debug_render_population(population, policy, duration, generation=0):
    """Open a window showing all population agents driving simultaneously."""
    import highway_env as _h  # noqa
    pop_size = len(population)
    print(f"\n  {C['cyan']}Opening debug render — {pop_size} agents, gen {generation}...{C['reset']}")
    print(f"  {C['dim']}Close the window to continue training.{C['reset']}")

    ma_config = {
        **ENV_CONFIG,
        "duration": duration,
        "controlled_vehicles": pop_size,
        # Prevent episode from terminating on crash — we handle it manually
        "offroad_terminal": False,
        **MA_ENV_CONFIG_EXTRA,
    }
    env = gym.make("highway-v0", render_mode="human")
    env.unwrapped.configure(ma_config)
    obs_tuple, _ = env.reset(seed=SEED)
    u = env.unwrapped
    u._is_terminated = lambda: False   # crashes don't end the episode
    _patch_agent_collisions(u)              # agents are invisible to each other
    _align_agents_same_start(u)
    obs_tuple = u.observation_type.observe()

    ep_rewards = np.zeros(pop_size)
    ep_crashed = np.zeros(pop_size, dtype=bool)
    npcs_debug = [v for v in u.road.vehicles if v not in set(u.controlled_vehicles)]

    for step in range(duration):
        actions = tuple(policy.act(obs_tuple[i], population[i]) for i in range(pop_size))
        obs_tuple, _, term, trunc, _ = env.step(actions)

        # Keep NPCs alive — they are indestructible background traffic
        for npc in npcs_debug:
            npc.crashed = False

        vehicles = u.controlled_vehicles
        for i in range(min(pop_size, len(vehicles))):
            v = vehicles[i]
            if not ep_crashed[i]:
                ep_rewards[i] += _per_vehicle_reward(u, actions[i], v)
                if v.crashed:
                    ep_crashed[i] = True
                    _park_crashed_agent(v)

        # Only stop when duration is reached — not on crash
        if trunc:
            break

    env.close()
    print(f"  Best agent reward this render: {ep_rewards.max():.2f}  "
          f"Crashes: {ep_crashed.sum()}/{pop_size}\n")

# ══════════════════════════════════════════════════════════════════
# Main GA
# ══════════════════════════════════════════════════════════════════
def run_ga(cfg):
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    is_linux  = platform.system() == "Linux"
    n_workers = cfg["n_workers"] if is_linux else 1

    env_tmp   = make_env()
    obs_dim   = int(np.prod(env_tmp.observation_space.shape))
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    policy = MLPPolicy(obs_dim, n_actions, cfg["hidden_sizes"])
    population   = rng.standard_normal((cfg["population_size"], policy.n_params)) * 0.1
    n_elite      = max(1, int(cfg["population_size"] * cfg["elite_frac"]))
    mutation_std = cfg["mutation_std"]

    log_dir = os.path.expanduser("~/tb_logs/simple_ga")
    os.makedirs(log_dir, exist_ok=True)
    writer  = SummaryWriter(log_dir=log_dir)

    print_header(cfg, policy, n_workers, is_linux)
    print(f"\n  {C['dim']}TensorBoard: tensorboard --logdir ~/tb_logs/simple_ga{C['reset']}\n")

    best_ever_fitness = -np.inf
    best_ever_params  = None
    gen_times         = []

    for gen in range(cfg["n_generations"]):
        t0 = time.time()

        pop_metrics = evaluate_population(
            population, policy,
            n_episodes=cfg["n_eval_episodes"],
            duration=cfg["train_duration"],
        )

        fitnesses = np.array([m["fitness"] for m in pop_metrics])
        elite_idx = np.argsort(fitnesses)[-n_elite:][::-1]
        best_idx  = elite_idx[0]
        best_m    = pop_metrics[best_idx]

        improved = fitnesses[best_idx] > best_ever_fitness
        if improved:
            best_ever_fitness = fitnesses[best_idx]
            best_ever_params  = population[best_idx].copy()

        elapsed = time.time() - t0
        gen_times.append(elapsed)
        eta = np.mean(gen_times[-10:]) * (cfg["n_generations"] - gen - 1)

        print_gen(gen, cfg["n_generations"], fitnesses, best_m,
                  mutation_std, elapsed, eta, best_ever_fitness, improved)
        log_tb(writer, gen, pop_metrics, best_m, mutation_std, elapsed)

        if cfg["render_every_n_gen"] > 0 and (gen+1) % cfg["render_every_n_gen"] == 0:
            debug_render_population(population, policy,
                                    duration=cfg["eval_duration"],
                                    generation=gen+1)

        # Breed
        new_pop = [population[i].copy() for i in elite_idx]
        while len(new_pop) < cfg["population_size"]:
            p_a   = tournament_select(population, fitnesses, cfg["tournament_size"])
            p_b   = tournament_select(population, fitnesses, cfg["tournament_size"])
            child = mutate(crossover(p_a, p_b, cfg["crossover_prob"]), mutation_std)
            new_pop.append(child)

        population   = np.array(new_pop)
        mutation_std *= cfg["mutation_decay"]

    # ── Final ──────────────────────────────────────────────
    total = sum(gen_times)
    print(f"\n{C['bold']}{'═'*72}{C['reset']}")
    print(f"{C['bold']}  Done!{C['reset']}  "
          f"Best fitness: {C['bold']}{C['green']}{best_ever_fitness:.2f}{C['reset']}  "
          f"Total time: {int(total//60)}m{int(total%60):02d}s  "
          f"Avg/gen: {np.mean(gen_times):.1f}s")

    save_path = os.path.join(log_dir, "best_agent.npy")
    np.save(save_path, best_ever_params)
    print(f"  Weights → {save_path}")
    print(f"{'═'*72}\n")

    print(f"{C['cyan']}Final evaluation (3 episodes, {cfg['eval_duration']} steps)...{C['reset']}")
    final = evaluate_individual(best_ever_params, policy, n_episodes=3,
                                render=True, duration=cfg["eval_duration"])
    print(f"\n  Reward  : {final['mean_reward']:.2f} ± {final['std_reward']:.2f}")
    print(f"  Crashes : {final['collision_rate']:.0%}")
    print(f"  Speed   : {final['mean_speed']:.1f} m/s  (std {final['std_speed']:.1f})")
    print(f"  LCs/ep  : {final['mean_lane_changes']:.1f}")
    print(f"  Actions : " + "  ".join(
        f"{n}:{v:.0%}" for n, v in zip(ACTION_NAMES, final["action_dist"])))

    writer.close()


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import multiprocessing as _mp
    p = argparse.ArgumentParser(
        description="Simple GA · highway-env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pop",          type=int,   default=GA_CONFIG["population_size"])
    p.add_argument("--gens",         type=int,   default=GA_CONFIG["n_generations"])
    p.add_argument("--episodes",     type=int,   default=GA_CONFIG["n_eval_episodes"])
    p.add_argument("--render-every", type=int,   default=GA_CONFIG["render_every_n_gen"],
                   help="Show all agents every N gens (0 = never)")
    p.add_argument("--debug", action="store_true",
                   help="Render one episode of the initial population then exit")
    p.add_argument("--mutation-std", type=float, default=GA_CONFIG["mutation_std"])
    p.add_argument("--train-dur",    type=int,   default=GA_CONFIG["train_duration"])
    p.add_argument("--workers",      type=int,   default=_mp.cpu_count(),
                   help="Parallel workers (Linux only, ignored on macOS)")
    args = p.parse_args()

    cfg = GA_CONFIG.copy()
    cfg["population_size"]    = args.pop
    cfg["n_generations"]      = args.gens
    cfg["n_eval_episodes"]    = args.episodes
    cfg["render_every_n_gen"] = args.render_every
    cfg["mutation_std"]       = args.mutation_std
    cfg["train_duration"]     = args.train_dur
    cfg["n_workers"]          = args.workers

    if args.debug:
        # Quick visual check: show all agents on screen then exit
        import gymnasium as gym
        import highway_env as _h  # noqa
        env_tmp = make_env()
        obs_dim   = int(np.prod(env_tmp.observation_space.shape))
        n_actions = env_tmp.action_space.n
        env_tmp.close()
        policy = MLPPolicy(obs_dim, n_actions, cfg["hidden_sizes"])
        rng = np.random.default_rng(SEED)
        pop = rng.standard_normal((cfg["population_size"], policy.n_params)) * 0.1
        debug_render_population(pop, policy, duration=cfg["eval_duration"], generation=0)
    else:
        run_ga(cfg)