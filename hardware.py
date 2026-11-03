"""Hypothetical hardware cost: operations per time step and the wire needed to lay a reservoir out on a chip.

Every reservoir is placed on the same 201 sites: the web's rings and spokes,
1 unit apart, inside a circular die of radius RINGS + 0.5. Each thread is a
straight wire between its two neurons; each input neuron also needs a wire
from the nearest point on the die's edge. Since a reservoir need not keep the
web's layout, both are also given an optimized placement that shuffles neurons
between sites to make the wiring as short as possible.
"""
import numpy as np
from topology import LAYOUT, RINGS, UNITS

DIE_RADIUS = RINGS + 0.5


def operations(W, Win):
    """Multiply-adds per time step: one per thread, one per input connection, one per readout weight (+ intercept)."""
    return {"threads": W.nnz, "input": Win.nnz, "readout": UNITS + 1, "total": W.nnz + Win.nnz + UNITS + 1}


def wiring(W, Win, slots):
    """Wire lengths when neuron i sits at site slots[i]."""
    xy = LAYOUT[slots]
    targets, sources = W.nonzero()
    threads = np.linalg.norm(xy[targets] - xy[sources], axis=1)
    inputs = DIE_RADIUS - np.linalg.norm(xy[np.unique(Win.nonzero()[0])], axis=1)
    return threads, inputs


def optimize_placement(W, Win, rng, swaps=200_000):
    """Greedy local search: swap two neurons' sites whenever that shortens the total wire."""
    counts = (W != 0).astype(int)
    counts = (counts + counts.T).tocsr()  # wires between i and j, either direction
    neighbours = [(counts.indices[counts.indptr[i] : counts.indptr[i + 1]],
                   counts.data[counts.indptr[i] : counts.indptr[i + 1]]) for i in range(UNITS)]
    has_input = np.zeros(UNITS, bool)
    has_input[Win.nonzero()[0]] = True
    radius = np.linalg.norm(LAYOUT, axis=1)
    slots = np.arange(UNITS)

    def cost_change(a, b):
        """Change in a's wiring if it moved from its site to b's (wires to b, or a self-loop, do not change)."""
        nbrs, n = neighbours[a]
        keep = (nbrs != b) & (nbrs != a)
        others = LAYOUT[slots[nbrs[keep]]]
        old, new = LAYOUT[slots[a]], LAYOUT[slots[b]]
        change = (n[keep] * (np.linalg.norm(others - new, axis=1) - np.linalg.norm(others - old, axis=1))).sum()
        return change + has_input[a] * (radius[slots[a]] - radius[slots[b]])

    for a, b in rng.integers(UNITS, size=(swaps, 2)):
        if a != b and cost_change(a, b) + cost_change(b, a) < 0:
            slots[a], slots[b] = slots[b], slots[a]
    return slots


def hardware_cost(W, Win, rng):
    natural = wiring(W, Win, np.arange(UNITS))
    optimized = wiring(W, Win, optimize_placement(W, Win, rng))
    return {
        **{f"ops_{k}": v for k, v in operations(W, Win).items()},
        "wire_natural": float(natural[0].sum() + natural[1].sum()),
        "wire_optimized": float(optimized[0].sum() + optimized[1].sum()),
        "input_wire_natural": float(natural[1].sum()),
        "longest_thread_natural": float(natural[0].max()),
        "thread_lengths_natural": natural[0].round(3).tolist(),
    }
