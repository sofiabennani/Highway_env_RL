# Genetic — neuroevolution on custom-highway-v0

All commands are run from the **project root** (`Highway_env_RL/`).
Results are saved in `genetic/results/<exp-name>/`.

---

## GA — Genetic Algorithm (single-objective)

```bash
python -m genetic.ga_highway
python -m genetic.ga_highway --pop 50 --gens 100 --workers 8
```

| Flag | Default | Description |
|---|---|---|
| `--pop` | `50` | Population size |
| `--gens` | `100` | Number of generations |
| `--episodes` | `3` | Episodes per individual |
| `--mutation-std` | `0.10` | Mutation standard deviation |
| `--train-dur` | `100` | Episode duration during training |
| `--render-every` | `10` | Render every N generations (0 = never) |
| `--workers` | cpu | Parallel workers |
| `--debug` | off | Verbose debug output |

---

## CMA-ES (single-objective)

```bash
# Train
python -m genetic.cmaeshw --exp-name my_run --generations 150

# Resume
python -m genetic.cmaeshw --exp-name my_run --resume

# Evaluate
python -m genetic.cmaeshw --exp-name my_run --evaluate
```

| Flag | Default | Description |
|---|---|---|
| `--exp-name` | `exp_01` | Output folder name |
| `--generations` | `150` | Max generations |
| `--sigma0` | `0.5` | Initial step size |
| `--popsize` | auto | Population size (≈ 4 + 3·ln(n_params)) |
| `--rollouts` | `5` | Episodes per individual |
| `--hidden` | `16` | MLP hidden layer size |
| `--workers` | cpu−1 | Parallel workers |
| `--resume` | off | Resume from checkpoint |
| `--evaluate` | off | Evaluate instead of train |

---

## NSGA-II (multi-objective)

```bash
# Train — time budget
python -m genetic.nsga2_highway --exp-name my_run --hours 7

# Train — generation budget
python -m genetic.nsga2_highway --exp-name my_run --generations 300

# Resume from last checkpoint
python -m genetic.nsga2_highway --exp-name my_run --resume --hours 5

# Evaluate saved Pareto front
python -m genetic.nsga2_highway --exp-name my_run --evaluate
```

| Flag | Default | Description |
|---|---|---|
| `--exp-name` | `nsga2_01` | Output folder name |
| `--hours` | `7.0` | Time budget (overrides `--generations`) |
| `--generations` | — | Max generations |
| `--popsize` | `50` | Population size |
| `--rollouts` | `5` | Episodes per individual |
| `--hidden` | `16` | MLP hidden layer size |
| `--workers` | cpu−1 | Parallel workers |
| `--resume` | off | Resume from checkpoint |
| `--evaluate` | off | Evaluate instead of train |

---

## Render a trained policy

```bash
# CMA-ES — best policy
python -m genetic.render_agent --exp-name my_run

# NSGA-II — list Pareto front then pick
python -m genetic.render_agent --exp-name my_run --nsga2 --list
python -m genetic.render_agent --exp-name my_run --nsga2 --policy-index 3
python -m genetic.render_agent --exp-name my_run --nsga2 --select safest
```

| Flag | Description |
|---|---|
| `--exp-name` | Experiment to load |
| `--nsga2` | Load from NSGA-II Pareto front (default: CMA-ES) |
| `--select` | Auto-pick `safest` / `fastest` / `balanced` |
| `--policy-index` | Pick by index in the Pareto front |
| `--list` | Print the Pareto front and exit |
| `--episodes` | Number of episodes to render (default: 3) |
| `--vehicles-density` | Traffic density (default: 1.0) |
| `--random` | Run a random policy as baseline |
