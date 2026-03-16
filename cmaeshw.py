"""
cmaes_highway.py
================
CMA-ES neuroevolution for "custom-highway-v0".

Evolves the weights of a small MLP policy. Fitness is averaged over several
rollouts for stability, and evaluation is parallelised across CPU cores.
Put custom_env.py in the same directory — it is imported to register the env.

Usage
-----
# Basic run
python cmaes_highway.py

# Named experiment
python cmaes_highway.py --exp-name run_01 --generations 150 --rollouts 5

# Larger network, denser traffic
python cmaes_highway.py --exp-name dense_32 --hidden 32 --vehicles-density 2.0

# Resume an interrupted run
python cmaes_highway.py --exp-name run_01 --resume

# Evaluate a saved policy (renders 20 episodes, prints stats)
python cmaes_highway.py --exp-name run_01 --evaluate

Output layout
-------------
results/
  <exp_name>/
    config.json       full config for reproducibility
    best_policy.npz   weights of the best individual seen
    final_policy.npz  weights at end of last generation
    checkpoint.pkl    full CMA-ES state (for --resume)
    log.csv           per-generation: mean, best, sigma, elapsed
"""

import argparse
import csv
import json
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import cma
import gymnasium as gym
import numpy as np

import custom_env  # noqa: F401 — registers "custom-highway-v0"
from custom_env import MLP, OBS_DIM, N_ACTIONS


# ---------------------------------------------------------------------------
# Rollout (runs in a subprocess — all imports must be inside)
# ---------------------------------------------------------------------------

def rollout(args) -> tuple:
    """Single episode. Returns (total_reward, crashed)."""
    weights, hidden_dim, vehicles_density, seed = args

    import gymnasium as gym
    import numpy as np
    from custom_env import MLP, OBS_DIM, N_ACTIONS  # also re-registers the env

    # Override only vehicles_density — all other params come from default_config()
    env = gym.make("custom-highway-v0", config={"vehicles_density": vehicles_density})
    policy = MLP(OBS_DIM, hidden_dim, N_ACTIONS)

    obs, _ = env.reset(seed=seed)
    total, done, truncated = 0.0, False, False
    info = {}

    while not (done or truncated):
        action = policy.forward(obs, weights)
        obs, reward, done, truncated, info = env.step(action)
        total += reward

    env.close()
    return total, bool(info.get("crashed", False))


def evaluate_weights(
    weights: np.ndarray,
    hidden_dim: int,
    n_rollouts: int,
    vehicles_density: float,
    pool: Pool,
    base_seed: int = 0,
) -> tuple:
    """
    Average reward and crash rate over n_rollouts episodes.
    Returns (mean_reward, crash_rate) where crash_rate is in [0, 1].
    """
    args = [
        (weights, hidden_dim, vehicles_density, base_seed + i)
        for i in range(n_rollouts)
    ]
    results = pool.map(rollout, args)
    rewards = [r for r, _ in results]
    crashes = [c for _, c in results]
    return float(np.mean(rewards)), float(np.mean(crashes))


# ---------------------------------------------------------------------------
# Experiment I/O
# ---------------------------------------------------------------------------

def exp_dir(exp_name: str) -> Path:
    path = Path("results") / exp_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_policy(directory: Path, weights: np.ndarray, filename: str):
    np.savez(directory / filename, weights=weights)
    print(f"  Saved {filename}")


def load_policy(directory: Path, filename: str = "best_policy.npz") -> np.ndarray:
    return np.load(directory / filename)["weights"]


def save_checkpoint(directory: Path, es, best_weights, best_fitness):
    with open(directory / "checkpoint.pkl", "wb") as f:
        pickle.dump(
            {"es": es, "best_weights": best_weights, "best_fitness": best_fitness}, f
        )


def load_checkpoint(directory: Path):
    with open(directory / "checkpoint.pkl", "rb") as f:
        d = pickle.load(f)
    return d["es"], d["best_weights"], d["best_fitness"]


def init_log(directory: Path) -> Path:
    path = directory / "log.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["generation", "mean_fitness", "best_fitness", "sigma", "elapsed_s", "crash_rate"]
        )
    return path


def append_log(path: Path, gen, mean_f, best_f, sigma, elapsed, crash_rate=0.0):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(
            [gen, f"{mean_f:.4f}", f"{best_f:.4f}", f"{sigma:.6f}", f"{elapsed:.1f}", f"{crash_rate:.4f}"]
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict):
    directory = exp_dir(config["exp_name"])

    # Save config for reproducibility
    with open(directory / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    hidden_dim = config["hidden"]
    policy     = MLP(OBS_DIM, hidden_dim, N_ACTIONS)
    n_params   = policy.n_params
    print(f"\nPolicy: {OBS_DIM} → {hidden_dim} → {N_ACTIONS}  ({n_params} params)")

    # CMA-ES — fresh or resumed
    if config["resume"] and (directory / "checkpoint.pkl").exists():
        print("Resuming from checkpoint...")
        es, best_weights, best_fitness = load_checkpoint(directory)
        start_gen = es.result.iterations
        log_path  = directory / "log.csv"
    else:
        es = cma.CMAEvolutionStrategy(
            np.zeros(n_params),
            config["sigma0"],
            {
                "popsize": config["popsize"] or (4 + int(3 * np.log(n_params))),
                "maxiter": config["generations"],
                "tolx":    1e-6,
                "tolfun":  1e-6,
                "verbose": -9,
            },
        )
        best_weights = np.zeros(n_params)
        best_fitness = -np.inf
        start_gen    = 0
        log_path     = init_log(directory)

    n_workers = config["workers"] or max(1, os.cpu_count() - 1)
    print(f"Workers: {n_workers}  |  Pop size: {es.popsize}  |  "
          f"Rollouts/individual: {config['rollouts']}  |  "
          f"Generations: {config['generations']}\n")

    t_start = time.time()

    with Pool(processes=n_workers) as pool:
        generation = start_gen
        while not es.stop() and generation < config["generations"]:
            generation += 1
            t0        = time.time()
            solutions = es.ask()

            fitnesses_and_crashes = [
                evaluate_weights(
                    w,
                    hidden_dim,
                    config["rollouts"],
                    config["vehicles_density"],
                    pool,
                    base_seed=generation * 1000,
                )
                for w in solutions
            ]

            fitnesses   = [f for f, _ in fitnesses_and_crashes]
            crash_rates = [c for _, c in fitnesses_and_crashes]

            es.tell(solutions, [-f for f in fitnesses])  # CMA-ES minimises

            gen_best_idx  = int(np.argmax(fitnesses))
            gen_best      = fitnesses[gen_best_idx]
            mean_fitness  = float(np.mean(fitnesses))
            mean_crashes  = float(np.mean(crash_rates))   # avg across population

            if gen_best > best_fitness:
                best_fitness = gen_best
                best_weights = solutions[gen_best_idx].copy()
                save_policy(directory, best_weights, "best_policy.npz")

            elapsed = time.time() - t_start
            print(
                f"Gen {generation:4d}/{config['generations']}  "
                f"mean: {mean_fitness:7.3f}  "
                f"best: {gen_best:7.3f}  "
                f"all-time: {best_fitness:7.3f}  "
                f"crashes: {mean_crashes * 100:5.1f}%  "
                f"σ: {es.sigma:.4f}  "
                f"({time.time() - t0:.1f}s)"
            )
            append_log(log_path, generation, mean_fitness, gen_best, es.sigma, elapsed, mean_crashes)

            if generation % 10 == 0:
                save_checkpoint(directory, es, best_weights, best_fitness)
                save_policy(directory, best_weights, "final_policy.npz")

    save_policy(directory, best_weights, "best_policy.npz")
    save_policy(directory, best_weights, "final_policy.npz")
    save_checkpoint(directory, es, best_weights, best_fitness)
    print(f"\nDone. Best fitness: {best_fitness:.4f}")
    print(f"Results in: {directory}/")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config: dict, n_episodes: int = 20):
    directory = exp_dir(config["exp_name"])
    weights   = load_policy(directory)
    policy    = MLP(OBS_DIM, config["hidden"], N_ACTIONS)
    env       = gym.make(
        "custom-highway-v0",
        config={"vehicles_density": config["vehicles_density"]},
    )

    print(f"\nEvaluating best policy — {n_episodes} episodes")
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total, done, truncated = 0.0, False, False
        while not (done or truncated):
            action = policy.forward(obs, weights)
            obs, reward, done, truncated, _ = env.step(action)
            total += reward
        rewards.append(total)
        print(f"  Episode {ep + 1:3d}: {total:.3f}")

    env.close()
    rewards = np.array(rewards)
    print(f"\nMean ± Std : {rewards.mean():.3f} ± {rewards.std():.3f}")
    print(f"Min  / Max : {rewards.min():.3f} / {rewards.max():.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CMA-ES neuroevolution — custom-highway-v0")

    p.add_argument("--exp-name",         type=str,   default="exp_01",
                   help="Experiment name — determines output folder (default: exp_01)")
    p.add_argument("--resume",           action="store_true",
                   help="Resume training from the latest checkpoint")
    p.add_argument("--evaluate",         action="store_true",
                   help="Evaluate saved best_policy.npz instead of training")

    p.add_argument("--hidden",           type=int,   default=16,
                   help="Hidden layer size (default: 16)")

    p.add_argument("--generations",      type=int,   default=150,
                   help="Maximum generations (default: 150)")
    p.add_argument("--sigma0",           type=float, default=0.5,
                   help="Initial CMA-ES step size (default: 0.5)")
    p.add_argument("--popsize",          type=int,   default=None,
                   help="Population size (default: auto ≈ 4 + 3·ln(n_params))")

    p.add_argument("--rollouts",         type=int,   default=5,
                   help="Rollouts per individual for stable fitness (default: 5)")

    p.add_argument("--vehicles-density", type=float, default=1.0,
                   help="Traffic density passed to the env (default: 1.0)")

    p.add_argument("--workers",          type=int,   default=None,
                   help="Parallel workers (default: cpu_count − 1)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    args   = parse_args()
    config = vars(args)

    if args.evaluate:
        evaluate(config)
    else:
        train(config)