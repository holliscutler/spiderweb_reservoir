"""Tune both reservoirs on validation data, score 20 fresh reservoirs of each on test data, and cost them.

Writes results/results.json. Run `python figures.py` afterwards for the figures and summary.
"""
import os

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"

import itertools
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from hardware import hardware_cost
from tasks import MEMORY, TEST, TRAIN, VAL, WARMUP, build_tasks
from topology import RESERVOIRS, build, weights

SPECTRAL_RADII = [0.5, 0.7, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
LEAK_RATES = [0.1, 0.3, 0.6, 1.0]
INPUT_SCALINGS = [0.1, 0.5, 1.0]
RIDGES = [1e-8, 1e-6, 1e-4, 1e-2]
TUNE_SEEDS = range(3)
TEST_SEEDS = range(100, 120)
EXAMPLE = ("Mackey-Glass, 100 steps", TEST_SEEDS[0], 500)  # task, seed, steps of test predictions to keep
VAL_SPLIT = slice(TRAIN, TRAIN + VAL)
TEST_SPLIT = slice(TRAIN + VAL, TRAIN + VAL + TEST)
OUT = Path(__file__).parent / "results"


def fit_readouts(x, y, ridges):
    """Closed-form ridge regression with an intercept, sharing the normal equations across ridges."""
    X = np.column_stack([np.ones(len(x)), x])[WARMUP:TRAIN]
    XtX, XtY = X.T @ X, X.T @ y[WARMUP:TRAIN]
    return [np.linalg.solve(XtX + r * np.eye(len(XtX)), XtY) for r in ridges]


def evaluate(task, x, y, readout, split):
    """(score, predictions). Forecasts score NRMSE; memory scores the summed r^2 over delays."""
    pred = np.column_stack([np.ones(len(x)), x])[split] @ readout
    true = y[split]
    if task == MEMORY:
        return float(sum(np.corrcoef(pred[:, k], true[:, k])[0, 1] ** 2 for k in range(true.shape[1]))), pred
    return float(np.sqrt(np.mean((pred - true) ** 2)) / true.std()), pred


def tune_one(task, name, seed, sr, lr, input_scaling, u, y):
    x = build(name, seed, sr, lr, input_scaling).run(u)
    return [((sr, lr, input_scaling, r), evaluate(task, x, y, w, VAL_SPLIT)[0])
            for r, w in zip(RIDGES, fit_readouts(x, y, RIDGES))]


def test_one(task, name, seed, params, u, y):
    sr, lr, input_scaling, ridge = params
    x = build(name, seed, sr, lr, input_scaling).run(u)
    return evaluate(task, x, y, fit_readouts(x, y, [ridge])[0], TEST_SPLIT)


def cost_one(name, seed):
    return hardware_cost(*weights(name, seed), np.random.default_rng(seed))


def main():
    start = time.time()
    results = {"tasks": {}, "example": {}, "hardware": {}}
    grid = list(itertools.product(SPECTRAL_RADII, LEAK_RATES, INPUT_SCALINGS))
    with Parallel(n_jobs=-1) as parallel:
        for task, (u, y) in build_tasks().items():
            results["tasks"][task] = {}
            sign = -1 if task == MEMORY else 1
            for name in RESERVOIRS:
                by_params = {}
                runs = parallel(delayed(tune_one)(task, name, s, *cfg, u, y) for cfg in grid for s in TUNE_SEEDS)
                for params, score in itertools.chain.from_iterable(runs):
                    by_params.setdefault(params, []).append(score)
                # A NaN validation score means the configuration diverged: never select it.
                best = min(by_params, key=lambda p: sign * np.mean(by_params[p]) if np.isfinite(by_params[p]).all()
                           else np.inf)
                tests = parallel(delayed(test_one)(task, name, s, best, u, y) for s in TEST_SEEDS)
                results["tasks"][task][name] = {
                    "params": dict(zip(("spectral_radius", "leak_rate", "input_scaling", "ridge"), best)),
                    "validation": float(np.mean(by_params[best])),
                    "test": [score for score, _ in tests],
                }
                if task == EXAMPLE[0]:
                    results["example"]["truth"] = y[TEST_SPLIT][: EXAMPLE[2], 0].round(5).tolist()
                    results["example"][name] = tests[0][1][: EXAMPLE[2], 0].round(5).tolist()
                print(f"{task:25s} {name:15s} test={np.mean(results['tasks'][task][name]['test']):.4f} "
                      f"params={best}", flush=True)
        for name in RESERVOIRS:
            costs = parallel(delayed(cost_one)(name, s) for s in TEST_SEEDS)
            results["hardware"][name] = {
                "per_seed": [{k: v for k, v in c.items() if k != "thread_lengths_natural"} for c in costs],
                "thread_lengths_example": costs[0]["thread_lengths_natural"],
            }
            print(f"hardware {name:15s} wire={np.mean([c['wire_natural'] for c in costs]):.0f} "
                  f"optimized={np.mean([c['wire_optimized'] for c in costs]):.0f}", flush=True)
    results["example"].update(task=EXAMPLE[0], seed=EXAMPLE[1])
    results["protocol"] = {
        "train": TRAIN, "validation": VAL, "test": TEST, "warmup": WARMUP,
        "spectral_radii": SPECTRAL_RADII, "leak_rates": LEAK_RATES, "input_scalings": INPUT_SCALINGS,
        "ridges": RIDGES, "tune_seeds": list(TUNE_SEEDS), "test_seeds": list(TEST_SEEDS),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT / 'results.json'} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
