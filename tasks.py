"""Benchmark tasks: three forecasting problems and the memory capacity test."""
import numpy as np
from reservoirpy.datasets import mackey_glass, santafe_laser

TRAIN, VAL, TEST, WARMUP = 4000, 1000, 1000, 100
TOTAL = TRAIN + VAL + TEST
MEMORY = "Memory capacity"
MEMORY_DELAYS = 300


def _forecast(series, horizon):
    return series[:TOTAL], series[horizon : TOTAL + horizon]


def _normalize(u, y):
    """Scale with training statistics only."""
    lo, hi = u[:TRAIN].min(), u[:TRAIN].max()
    return 2 * (u - lo) / (hi - lo) - 1, (y - y[:TRAIN].mean(0)) / y[:TRAIN].std(0)


def build_tasks():
    """name -> (input, target), each of length TOTAL."""
    tasks = {
        "Mackey-Glass, 20 steps": _forecast(mackey_glass(TOTAL + 20, seed=0), 20),
        "Mackey-Glass, 100 steps": _forecast(mackey_glass(TOTAL + 100, seed=0), 100),
        "Santa Fe laser, 1 step": _forecast(santafe_laser()[: TOTAL + 1].astype(float), 1),
    }
    # Memory: recall the input from 1 to MEMORY_DELAYS steps ago.
    u = np.random.default_rng(0).uniform(-0.8, 0.8, size=(TOTAL + MEMORY_DELAYS, 1))
    lags = np.column_stack([u[MEMORY_DELAYS - k : -k, 0] for k in range(1, MEMORY_DELAYS + 1)])
    tasks[MEMORY] = (u[MEMORY_DELAYS:], lags)
    return {name: _normalize(*task) for name, task in tasks.items()}
