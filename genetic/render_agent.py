"""
render_agent.py
===============
Render trained policies from either CMA-ES or NSGA-II experiments.

CMA-ES usage
------------
# Best policy from a CMA-ES run
python render_agent.py --exp-name exp_01

# Specific saved file
python render_agent.py --exp-name exp_01 --policy final_policy.npz

NSGA-II usage
-------------
# List all policies on the Pareto front (no render)
python render_agent.py --exp-name night_run --nsga2 --list

# Render a specific policy by index (from the list above)
python render_agent.py --exp-name night_run --nsga2 --policy-index 3

# Render the safest policy (lowest crash rate on the front)
python render_agent.py --exp-name night_run --nsga2 --select safest

# Render the fastest policy
python render_agent.py --exp-name night_run --nsga2 --select fastest

# Render the best compromise (closest to utopia point)
python render_agent.py --exp-name night_run --nsga2 --select balanced

Other
-----
python render_agent.py --random    # random agent baseline
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import genetic.custom_env as custom_env  # noqa
from genetic.custom_env import MLP, OBS_DIM, N_ACTIONS


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_cmaes_policy(exp_name: str, filename: str) -> np.ndarray:
    path = Path(__file__).parent / "results" / exp_name / filename
    if not path.exists():
        raise FileNotFoundError(f"No policy found at {path}")
    return np.load(path)["weights"]


def load_nsga2_front(exp_name: str):
    """
    Returns (weights_list, objectives, hidden_dim) where hidden_dim may be
    None for old saves that didn't store architecture metadata.
    """
    directory  = Path(__file__).parent / "results" / exp_name
    front_path = directory / "pareto_front.npz"
    obj_path   = directory / "pareto_objectives.npy"

    if not front_path.exists():
        raise FileNotFoundError(
            f"No Pareto front at {front_path}\n"
            f"Run: python nsga2_highway.py --exp-name {exp_name}"
        )

    data         = np.load(front_path)
    weight_keys  = sorted(k for k in data.files if not k.startswith("_"))
    weights_list = [data[k] for k in weight_keys]
    objectives   = np.load(obj_path)
    hidden_dim   = int(data["_hidden_dim"]) if "_hidden_dim" in data.files else None
    return weights_list, objectives, hidden_dim


def select_policy(weights_list, objectives, mode: str, index: int = None):
    """Returns (weights, label, obj_row)."""
    if mode == "index":
        if index is None or index >= len(weights_list):
            raise ValueError(f"--policy-index must be in [0, {len(weights_list)-1}]")
        i = index
    elif mode == "safest":
        i = int(np.argmax(objectives[:, 1]))
    elif mode == "fastest":
        i = int(np.argmax(objectives[:, 0]))
    elif mode == "balanced":
        utopia  = objectives.max(axis=0)
        obj_min = objectives.min(axis=0)
        rng     = np.where(objectives.max(axis=0) - obj_min > 0,
                           objectives.max(axis=0) - obj_min, 1.0)
        norm    = (objectives - obj_min) / rng
        norm_u  = (utopia - obj_min) / rng
        i       = int(np.argmin(np.linalg.norm(norm - norm_u, axis=1)))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    obj   = objectives[i]
    label = f"{mode} (#{i})  speed={obj[0]:.1f}  safety={obj[1]:.1f}  smooth={obj[2]:.1f}"
    return weights_list[i], label, obj


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, policy_fn, seed: int):
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    done = truncated = False
    info = {}
    while not (done or truncated):
        action = policy_fn(obs)
        obs, reward, done, truncated, info = env.step(action)
        total += reward
        steps += 1
    return total, steps, bool(info.get("crashed", False))


# ---------------------------------------------------------------------------
# List Pareto front
# ---------------------------------------------------------------------------

def list_front(exp_name: str):
    weights_list, objectives, hidden_dim = load_nsga2_front(exp_name)
    n    = len(weights_list)
    arch = f"hidden={hidden_dim}" if hidden_dim is not None else "hidden=unknown"
    print(f"\nPareto front — {exp_name}  ({n} policies, {arch})\n")
    print(f"  {'#':>4}  {'speed':>8}  {'safety':>8}  {'smooth':>8}  profile")
    print("  " + "─" * 58)
    for i in np.argsort(-objectives[:, 1]):
        obj  = objectives[i]
        safe = "safe"   if obj[1] >= -1  else ("moderate" if obj[1] >= -5 else "risky")
        fast = "fast"   if obj[0] > 200  else ("medium"   if obj[0] > 100 else "slow")
        print(f"  {i:4d}  {obj[0]:8.2f}  {obj[1]:8.2f}  {obj[2]:8.2f}  {safe} + {fast}")
    print()
    print("  safety: 0 = no crash ever, -10 = crashed every episode")
    print("  Use --policy-index N  or  --select safest/fastest/balanced\n")


# ---------------------------------------------------------------------------
# Render loop
# ---------------------------------------------------------------------------

def render(policy_fn, label, args):
    print(f"\n{label}")
    print(f"Running {args.episodes} episode(s)\n")

    env = gym.make(
        "custom-highway-v0",
        config={"vehicles_density": args.vehicles_density},
        render_mode="human",
    )

    rewards, crashes = [], []

    for ep in range(args.episodes):
        total, steps, crashed = run_episode(env, policy_fn, seed=args.seed + ep)
        rewards.append(total)
        crashes.append(crashed)

        n_done      = ep + 1
        crash_count = sum(crashes)
        crash_pct   = 100 * crash_count / n_done
        status      = "CRASH" if crashed else "ok"

        print(
            f"  Episode {n_done:3d}/{args.episodes}"
            f" | reward: {total:8.3f}"
            f" | steps: {steps:4d}"
            f" | {status:<6}"
            f" | crashes: {crash_count}/{n_done} ({crash_pct:.0f}%)"
        )

        if ep < args.episodes - 1:
            time.sleep(1.0)

    env.close()

    rewards     = np.array(rewards)
    crash_count = sum(crashes)
    print(f"\n{'─' * 50}")
    print(f"  Episodes    : {args.episodes}")
    print(f"  Mean reward : {rewards.mean():.3f} ± {rewards.std():.3f}")
    print(f"  Min / Max   : {rewards.min():.3f} / {rewards.max():.3f}")
    print(f"  Crash rate  : {crash_count}/{args.episodes} ({100*crash_count/args.episodes:.0f}%)")
    print(f"{'─' * 50}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.nsga2:
        if args.list:
            list_front(args.exp_name)
            return

        weights_list, objectives, saved_hidden = load_nsga2_front(args.exp_name)

        # Use architecture from the saved file — never trust --hidden for NSGA-II
        hidden_dim = saved_hidden if saved_hidden is not None else args.hidden
        if saved_hidden is not None and saved_hidden != args.hidden:
            print(f"  Note: using hidden={hidden_dim} from saved file (not --hidden={args.hidden})")

        mlp       = MLP(OBS_DIM, hidden_dim, N_ACTIONS)
        mode      = "index" if args.policy_index is not None else args.select
        weights, label, _ = select_policy(weights_list, objectives, mode, args.policy_index)
        label     = f"NSGA-II — {args.exp_name} — {label}"
        policy_fn = lambda obs: mlp.forward(obs, weights)

    elif args.random:
        label     = "Random agent"
        policy_fn = lambda obs: np.random.randint(N_ACTIONS)

    else:
        # CMA-ES — architecture comes from --hidden (stored in config.json)
        mlp       = MLP(OBS_DIM, args.hidden, N_ACTIONS)
        weights   = load_cmaes_policy(args.exp_name, args.policy)
        policy_fn = lambda obs: mlp.forward(obs, weights)
        label     = f"CMA-ES — {args.exp_name}/{args.policy}"

    render(policy_fn, label, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Render CMA-ES or NSGA-II policies")

    p.add_argument("--exp-name",         type=str,   default="exp_01")
    p.add_argument("--hidden",           type=int,   default=16)
    p.add_argument("--episodes",         type=int,   default=5)
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--vehicles-density", type=float, default=1.0)
    p.add_argument("--random",           action="store_true")

    # CMA-ES
    p.add_argument("--policy",           type=str,   default="best_policy.npz")

    # NSGA-II
    p.add_argument("--nsga2",            action="store_true",
                   help="Load from NSGA-II Pareto front")
    p.add_argument("--list",             action="store_true",
                   help="[NSGA-II] Print the Pareto front and exit")
    p.add_argument("--policy-index",     type=int,   default=None,
                   help="[NSGA-II] Index of policy to render")
    p.add_argument("--select",           type=str,   default="balanced",
                   choices=["safest", "fastest", "balanced"],
                   help="[NSGA-II] Auto-select a policy (default: balanced)")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())