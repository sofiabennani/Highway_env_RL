"""
render_agent.py
===============
Visualise a trained CMA-ES policy on custom-highway-v0.

Usage
-----
# Watch the best policy from exp_01 (5 episodes)
python render_agent.py --exp-name exp_01

# More episodes, slower speed
python render_agent.py --exp-name exp_01 --episodes 10 --fps 15

# Try a specific saved policy (e.g. final_policy.npz)
python render_agent.py --exp-name exp_01 --policy final_policy.npz

# Random agent baseline for comparison
python render_agent.py --random
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import custom_env as custom_env  # noqa — registers "custom-highway-v0"
from custom_env import MLP, OBS_DIM, N_ACTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_policy(exp_name: str, filename: str) -> np.ndarray:
    path = Path("results") / exp_name / filename
    if not path.exists():
        raise FileNotFoundError(f"No policy found at {path}")
    return np.load(path)["weights"]


def run_episode(env, policy_fn, seed: int) -> tuple[float, int, bool]:
    """Run one episode. Returns (total_reward, n_steps, crashed)."""
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    done = truncated = False

    while not (done or truncated):
        action = policy_fn(obs)
        obs, reward, done, truncated, info = env.step(action)
        total += reward
        steps += 1

    crashed = info.get("crashed", False)
    return total, steps, crashed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Build policy function
    if args.random:
        label = "Random agent"
        policy_fn = lambda obs: np.random.randint(N_ACTIONS)
    else:
        weights = load_policy(args.exp_name, args.policy)
        mlp     = MLP(OBS_DIM, args.hidden, N_ACTIONS)
        policy_fn = lambda obs: mlp.forward(obs, weights)
        label = f"CMA-ES — {args.exp_name}/{args.policy}"

    print(f"\n{label}")
    print(f"Running {args.episodes} episode(s) at ~{args.fps} fps\n")

    env = gym.make(
        "custom-highway-v0",
        config={"vehicles_density": args.vehicles_density},
        render_mode="human",
    )

    rewards, steps_list, crashes = [], [], []

    for ep in range(args.episodes):
        total, steps, crashed = run_episode(env, policy_fn, seed=args.seed + ep)
        rewards.append(total)
        steps_list.append(steps)
        crashes.append(crashed)

        status = "CRASH" if crashed else "ok"
        print(f"  Episode {ep + 1:3d} | reward: {total:8.3f} | steps: {steps:4d} | {status}")

        # Pause between episodes so the window doesn't flash
        if ep < args.episodes - 1:
            time.sleep(1.0)

    env.close()

    # Summary
    rewards = np.array(rewards)
    print(f"\n{'─' * 45}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Mean reward: {rewards.mean():.3f} ± {rewards.std():.3f}")
    print(f"  Min / Max  : {rewards.min():.3f} / {rewards.max():.3f}")
    print(f"  Crash rate : {sum(crashes)}/{args.episodes}")
    print(f"{'─' * 45}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Render a trained CMA-ES policy")

    p.add_argument("--exp-name",         type=str,   default="exp_01",
                   help="Experiment folder under results/ (default: exp_01)")
    p.add_argument("--policy",           type=str,   default="best_policy.npz",
                   help="Policy file to load (default: best_policy.npz)")
    p.add_argument("--hidden",           type=int,   default=16,
                   help="Hidden layer size — must match training (default: 16)")

    p.add_argument("--episodes",         type=int,   default=5,
                   help="Number of episodes to render (default: 5)")
    p.add_argument("--fps",              type=int,   default=30,
                   help="Target render FPS — cosmetic only (default: 30)")
    p.add_argument("--seed",             type=int,   default=0,
                   help="Base random seed; each episode uses seed+i (default: 0)")

    p.add_argument("--vehicles-density", type=float, default=1.0,
                   help="Traffic density (default: 1.0)")

    p.add_argument("--random",           action="store_true",
                   help="Run a random agent instead of a trained policy")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())