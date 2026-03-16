"""
nsga2_highway.py
================
NSGA-II multi-objective neuroevolution for "custom-highway-v0".

Optimises three objectives simultaneously — no reward scalarisation:
  f1 = -mean(speed_term)         → maximise speed reward
  f2 = -mean(collision_term)     → minimise crashes   (negative = we minimise)
  f3 = -mean(smoothness_term)    → maximise smoothness (acceleration + jerk + stable)

NSGA-II (pymoo) handles selection, crossover, mutation and Pareto ranking.
Fitness evaluation is parallelised across CPU cores via multiprocessing.Pool.

Usage
-----
python nsga2_highway.py                              # defaults: 7h budget
python nsga2_highway.py --exp-name night_run --hours 7
python nsga2_highway.py --exp-name night_run --generations 300 --popsize 50
python nsga2_highway.py --exp-name night_run --evaluate  # evaluate saved front

Output
------
results/<exp_name>/
  config.json            full config for reproducibility
  pareto_front.npz       weights of all non-dominated policies found
  pareto_objectives.npy  objective vectors for each policy on the front
  log.csv                per-generation stats
  checkpoint.npz         full population + objectives (resume not yet supported)
"""

import argparse
import csv
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.gauss import GM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.termination.max_gen import MaximumGenerationTermination

import gymnasium as gym
import custom_env  # noqa — registers "custom-highway-v0"
from custom_env import MLP, OBS_DIM, N_ACTIONS

# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
# We expose 3 objectives to NSGA-II (all minimised — pymoo convention):
#   F[0] = -mean(speed_term)          higher speed reward → lower F[0]
#   F[1] = -mean(collision_term)      fewer crashes → less negative → lower F[1]
#                                     collision_term is already negative (−10 or 0)
#                                     so -collision_term is 0 or +10 → minimise
#   F[2] = -mean(smoothness_term)     smoothness = acceleration + jerk + stable_speed

SMOOTHNESS_KEYS = ["acceleration", "jerk", "stable_speed"]

# ---------------------------------------------------------------------------
# Rollout (subprocess-safe — top-level function)
# ---------------------------------------------------------------------------

def rollout(args) -> tuple:
    """
    Single episode. Returns (speed_sum, collision_sum, smoothness_sum, crashed).
    All values are episode totals (not per-step averages).
    """
    weights, hidden_dim, vehicles_density, seed = args

    import gymnasium as gym
    import numpy as np
    from custom_env import MLP, OBS_DIM, N_ACTIONS

    env = gym.make("custom-highway-v0", config={"vehicles_density": vehicles_density})
    policy = MLP(OBS_DIM, hidden_dim, N_ACTIONS)

    obs, _ = env.reset(seed=seed)
    done = truncated = False
    info = {}

    speed_sum      = 0.0
    collision_sum  = 0.0
    smoothness_sum = 0.0

    while not (done or truncated):
        action = policy.forward(obs, weights)
        obs, _, done, truncated, info = env.step(action)
        terms = info.get("reward_terms", {})
        speed_sum      += terms.get("speed", 0.0)
        collision_sum  += terms.get("collision", 0.0)
        smoothness_sum += sum(terms.get(k, 0.0) for k in SMOOTHNESS_KEYS)

    env.close()
    crashed = bool(info.get("crashed", False))
    return speed_sum, collision_sum, smoothness_sum, crashed


def evaluate_individual(args) -> tuple:
    """
    Average objectives over n_rollouts episodes for one individual.
    Returns (F1, F2, F3, crash_rate) where F-values are negated for minimisation.
    """
    weights, hidden_dim, n_rollouts, vehicles_density, base_seed = args

    import gymnasium as gym
    import numpy as np
    from custom_env import MLP, OBS_DIM, N_ACTIONS

    rollout_args = [
        (weights, hidden_dim, vehicles_density, base_seed + i)
        for i in range(n_rollouts)
    ]

    # Import Pool inside subprocess is fine — we use sequential eval here
    # because the outer Pool already parallelises across individuals
    results = [rollout(a) for a in rollout_args]

    speeds      = [r[0] for r in results]
    collisions  = [r[1] for r in results]
    smoothness  = [r[2] for r in results]
    crashes     = [r[3] for r in results]

    f1 = -float(np.mean(speeds))       # minimise → maximise speed
    f2 = -float(np.mean(collisions))   # collision_term ≤ 0, so -collision ≥ 0, minimise
    f3 = -float(np.mean(smoothness))   # minimise → maximise smoothness

    return f1, f2, f3, float(np.mean(crashes))


# ---------------------------------------------------------------------------
# pymoo Problem wrapper
# ---------------------------------------------------------------------------

class HighwayProblem(Problem):
    """
    Wraps the highway driving task as a 3-objective minimisation problem
    for pymoo's NSGA-II.
    """

    def __init__(self, hidden_dim, n_rollouts, vehicles_density, n_workers, generation_counter):
        n_params = MLP(OBS_DIM, hidden_dim, N_ACTIONS).n_params
        # Weights initialised in [-1, 1]; unconstrained after that
        super().__init__(
            n_var=n_params,
            n_obj=3,
            xl=-1.0,
            xu=1.0,
        )
        self.hidden_dim       = hidden_dim
        self.n_rollouts       = n_rollouts
        self.vehicles_density = vehicles_density
        self.n_workers        = n_workers
        self.generation_counter = generation_counter  # shared mutable list [gen]

    def _evaluate(self, X, out, *args, **kwargs):
        """
        X : (popsize, n_params) array — one row per individual.
        out["F"] : (popsize, 3) objective matrix.
        """
        gen = self.generation_counter[0]
        base_seed = gen * 10000

        eval_args = [
            (X[i], self.hidden_dim, self.n_rollouts,
             self.vehicles_density, base_seed + i * 100)
            for i in range(len(X))
        ]

        with Pool(processes=self.n_workers) as pool:
            results = pool.map(evaluate_individual, eval_args)

        F           = np.array([[r[0], r[1], r[2]] for r in results])
        crash_rates = np.array([r[3] for r in results])

        out["F"]           = F
        out["crash_rates"] = crash_rates  # stored for logging via callback


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class LogCallback(Callback):
    """Called after each generation — logs stats and prints to console."""

    def __init__(self, log_path: Path, t_start: float):
        super().__init__()
        self.log_path = log_path
        self.t_start  = t_start
        self.gen      = 0

    def notify(self, algorithm):
        self.gen += 1

        # Update generation counter in the problem (for seed diversity)
        algorithm.problem.generation_counter[0] = self.gen

        F           = algorithm.pop.get("F")
        crash_rates = algorithm.pop.get("crash_rates")

        # Pareto front of current population
        from pymoo.indicators.hv import HV
        ref_point = np.array([50.0, 10.5, 50.0])  # slightly worse than worst possible
        try:
            hv_val = HV(ref_point=ref_point)(F)
        except Exception:
            hv_val = float("nan")

        mean_f1    = float(np.mean(F[:, 0]))
        mean_f2    = float(np.mean(F[:, 1]))
        mean_f3    = float(np.mean(F[:, 2]))
        mean_crash = float(np.mean(crash_rates)) if crash_rates is not None else float("nan")
        elapsed    = time.time() - self.t_start

        # Count non-dominated individuals in current population
        n_pareto = int(np.sum(algorithm.pop.get("rank") == 0))

        print(
            f"Gen {self.gen:4d}  "
            f"HV: {hv_val:8.3f}  "
            f"speed: {-mean_f1:6.3f}  "
            f"safety: {-mean_f2:6.3f}  "
            f"smooth: {-mean_f3:6.3f}  "
            f"crashes: {mean_crash*100:5.1f}%  "
            f"pareto_size: {n_pareto:3d}  "
            f"({elapsed/3600:.2f}h)"
        )

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.gen,
                f"{hv_val:.6f}",
                f"{-mean_f1:.4f}",
                f"{-mean_f2:.4f}",
                f"{-mean_f3:.4f}",
                f"{mean_crash:.4f}",
                n_pareto,
                f"{elapsed:.1f}",
            ])


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def make_exp_dir(exp_name: str) -> Path:
    path = Path("results") / exp_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_front(directory: Path, weights_list, objectives):
    np.savez(directory / "pareto_front.npz",
             **{f"w_{i}": w for i, w in enumerate(weights_list)})
    np.save(directory / "pareto_objectives.npy", objectives)
    print(f"  Saved Pareto front: {len(weights_list)} policies")


def load_front(directory: Path):
    data = np.load(directory / "pareto_front.npz")
    weights_list = [data[f"w_{i}"] for i in range(len(data.files))]
    objectives   = np.load(directory / "pareto_objectives.npy")
    return weights_list, objectives


def init_log(directory: Path) -> Path:
    path = directory / "log.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "generation", "hypervolume",
            "mean_speed", "mean_safety", "mean_smoothness",
            "crash_rate", "pareto_size", "elapsed_s"
        ])
    return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict):
    directory = make_exp_dir(config["exp_name"])

    with open(directory / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    hidden_dim = config["hidden"]
    n_params   = MLP(OBS_DIM, hidden_dim, N_ACTIONS).n_params
    n_workers  = config["workers"] or max(1, os.cpu_count() - 1)
    popsize    = config["popsize"]

    print(f"\nNSGA-II — custom-highway-v0")
    print(f"Policy: {OBS_DIM} → {hidden_dim} → {N_ACTIONS}  ({n_params} params)")
    print(f"Population: {popsize}  |  Rollouts/individual: {config['rollouts']}")
    print(f"Workers: {n_workers}  |  Objectives: speed, safety, smoothness")
    if config["hours"]:
        print(f"Budget: {config['hours']}h time limit")
    else:
        print(f"Budget: {config['generations']} generations")
    print()

    generation_counter = [0]

    problem = HighwayProblem(
        hidden_dim        = hidden_dim,
        n_rollouts        = config["rollouts"],
        vehicles_density  = config["vehicles_density"],
        n_workers         = n_workers,
        generation_counter= generation_counter,
    )

    algorithm = NSGA2(
        pop_size   = popsize,
        sampling   = FloatRandomSampling(),
        crossover  = SBX(prob=0.9, eta=15),   # SBX: good for continuous weights
        mutation   = GM(prob=1/n_params, sigma=0.1),  # Gaussian mutation
        eliminate_duplicates=True,
    )

    # Termination: time limit OR max generations (whichever fires first)
    if config["hours"]:
        termination = TimeBasedTermination(config["hours"] * 3600)
    else:
        termination = MaximumGenerationTermination(config["generations"])

    t_start  = time.time()
    log_path = init_log(directory)
    callback = LogCallback(log_path, t_start)

    result = minimize(
        problem,
        algorithm,
        termination,
        callback  = callback,
        verbose   = False,
        seed      = config["seed"],
    )

    # Extract final Pareto front
    pareto_X = result.X   # (n_pareto, n_params)
    pareto_F = result.F   # (n_pareto, 3) — negated objectives

    weights_list = [pareto_X[i] for i in range(len(pareto_X))]
    # Convert back to readable form: [speed, -collision, smoothness]
    objectives_readable = np.column_stack([
        -pareto_F[:, 0],   # speed (higher = better)
        -pareto_F[:, 1],   # safety: -collision_term (higher = fewer crashes)
        -pareto_F[:, 2],   # smoothness (higher = better)
    ])

    save_front(directory, weights_list, objectives_readable)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/3600:.2f}h  |  Pareto front: {len(weights_list)} policies")
    print(f"Results in: {directory}/")

    # Print front summary
    print("\nPareto front summary (speed | safety | smoothness):")
    # Sort by safety descending
    order = np.argsort(-objectives_readable[:, 1])
    for i in order[:10]:   # top 10 safest
        s = objectives_readable[i]
        print(f"  Policy {i:3d}: speed={s[0]:7.2f}  safety={s[1]:7.2f}  smooth={s[2]:7.2f}")
    if len(weights_list) > 10:
        print(f"  ... ({len(weights_list)-10} more policies on the front)")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config: dict, n_episodes: int = 10):
    directory    = make_exp_dir(config["exp_name"])
    weights_list, objectives = load_front(directory)

    hidden_dim = config["hidden"]
    policy     = MLP(OBS_DIM, hidden_dim, N_ACTIONS)

    print(f"\nPareto front has {len(weights_list)} policies")
    print(f"Evaluating each over {n_episodes} episodes...\n")
    print(f"{'Policy':>6}  {'Speed':>7}  {'Safety':>8}  {'Smooth':>7}  {'Crash%':>7}")
    print("─" * 50)

    for i, weights in enumerate(weights_list):
        env = gym.make("custom-highway-v0",
                       config={"vehicles_density": config["vehicles_density"]})
        rewards, crashes = [], []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            total, done, truncated = 0.0, False, False
            info = {}
            while not (done or truncated):
                action = policy.forward(obs, weights)
                obs, reward, done, truncated, info = env.step(action)
                total += reward
            rewards.append(total)
            crashes.append(info.get("crashed", False))

        env.close()
        crash_pct = 100 * sum(crashes) / n_episodes
        obj = objectives[i]
        print(f"  {i:4d}   {obj[0]:7.2f}   {obj[1]:8.2f}   {obj[2]:7.2f}   {crash_pct:6.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="NSGA-II neuroevolution — custom-highway-v0")

    p.add_argument("--exp-name",         type=str,   default="nsga2_01",
                   help="Experiment name (default: nsga2_01)")
    p.add_argument("--evaluate",         action="store_true",
                   help="Evaluate saved Pareto front instead of training")

    p.add_argument("--hidden",           type=int,   default=16,
                   help="Hidden layer size (default: 16)")

    p.add_argument("--hours",            type=float, default=7.0,
                   help="Time budget in hours — overrides --generations (default: 7.0)")
    p.add_argument("--generations",      type=int,   default=None,
                   help="Max generations — ignored if --hours is set")
    p.add_argument("--popsize",          type=int,   default=50,
                   help="Population size (default: 50)")
    p.add_argument("--rollouts",         type=int,   default=5,
                   help="Rollouts per individual (default: 5)")
    p.add_argument("--seed",             type=int,   default=42,
                   help="NSGA-II random seed (default: 42)")

    p.add_argument("--vehicles-density", type=float, default=1.0,
                   help="Traffic density (default: 1.0)")
    p.add_argument("--workers",          type=int,   default=None,
                   help="CPU workers (default: cpu_count − 1)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    args   = parse_args()
    config = vars(args)

    # --generations without --hours: disable time limit
    if config["generations"] is not None:
        config["hours"] = None

    if args.evaluate:
        evaluate(config)
    else:
        train(config)