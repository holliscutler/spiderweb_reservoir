"""The spider web reservoir and the default random reservoir it is compared to.

The web has 201 neurons: a hub (node 0) and 10 rings x 20 spokes around it.
Node `1 + r * SPOKES + s` sits on ring `r` (0 = innermost) and spoke `s`.
Threads carry signals inward, from the rim towards the hub, and one anchor
thread carries the hub back out to the rim. Input enters only at the rim.
"""
import numpy as np
from reservoirpy.nodes import Reservoir
from scipy import sparse

RINGS, SPOKES = 10, 20
UNITS = 1 + RINGS * SPOKES
RIM = np.arange(1 + (RINGS - 1) * SPOKES, UNITS)
REWIRE_P = 0.1
WEB, DEFAULT = "Spider web", "Default random"
RESERVOIRS = (WEB, DEFAULT)


def position(node):
    """Where a neuron sits in the plane: rings are 1 unit apart."""
    if node == 0:
        return 0.0, 0.0
    r, s = divmod(node - 1, SPOKES)
    angle = 2 * np.pi * s / SPOKES
    return (r + 1) * np.cos(angle), (r + 1) * np.sin(angle)


LAYOUT = np.array([position(n) for n in range(UNITS)])


def web_threads(spiral_reach=5, ring_reach=4, spoke_reach=2):
    """Directed (source, target) threads of the unrewired web, all pointing inward.

    Each neuron feeds the next `spiral_reach` neurons inward along the spiral
    (node k to k - 1, ..., k - spiral_reach) and, on each of the next
    `ring_reach` rings inward, the neurons within `spoke_reach` spokes of its
    own. Targets past the innermost ring land on the hub. The anchor thread
    from the hub to the outermost neuron closes the loop: without it the graph
    has no cycles, so nothing echoes and the spectral radius is zero.
    """
    last = RINGS * SPOKES
    threads = set()
    for k in range(1, last + 1):
        r, s = divmod(k - 1, SPOKES)
        threads.update((k, max(k - d, 0)) for d in range(1, spiral_reach + 1))
        for dr in range(1, ring_reach + 1):
            for ds in range(-spoke_reach, spoke_reach + 1):
                threads.add((k, 0 if r - dr < 0 else 1 + (r - dr) * SPOKES + (s + ds) % SPOKES))
    threads.add((0, last))
    return sorted(threads)


WEB_THREADS = web_threads()


def rewire(threads, p, rng):
    """Watts-Strogatz style: each thread, with probability p, gets a random new target."""
    present = set(threads)
    out = []
    for a, b in threads:
        if rng.random() < p:
            for _ in range(100):
                c = int(rng.integers(UNITS))
                if c != a and (a, c) not in present:
                    present.discard((a, b))
                    present.add((a, c))
                    b = c
                    break
        out.append((a, b))
    return out


def web_matrix(sr, rng):
    """Recurrent weights W[target, source] of one web, scaled to spectral radius `sr`."""
    while True:  # rewiring can, rarely, break every loop; draw again
        sources, targets = np.array(rewire(WEB_THREADS, REWIRE_P, rng)).T
        W = sparse.coo_matrix((rng.normal(size=len(sources)), (targets, sources)), shape=(UNITS, UNITS))
        radius = np.abs(np.linalg.eigvals(W.toarray())).max()
        if radius > 1e-9:
            return sparse.csr_matrix(W * (sr / radius))


def build(name, seed, sr=1.0, lr=1.0, input_scaling=1.0):
    """A ReservoirPy reservoir. The default is ReservoirPy's own: 10% connectivity, 10% of neurons get input."""
    if name == DEFAULT:
        return Reservoir(units=UNITS, sr=sr, lr=lr, input_scaling=input_scaling, seed=seed)
    rng = np.random.default_rng(seed)
    W = web_matrix(sr, rng)
    # ReservoirPy uses a ready-made W and Win as given, so both are scaled here.
    Win = np.zeros((UNITS, 1))
    Win[RIM] = input_scaling * rng.choice([-1.0, 1.0], size=(len(RIM), 1))
    return Reservoir(units=UNITS, W=W, Win=Win, lr=lr, seed=seed)


def weights(name, seed):
    """(W, Win) of one reservoir at spectral radius 1, as sparse matrices."""
    reservoir = build(name, seed)
    reservoir.run(np.zeros((1, 1)))
    return sparse.csr_matrix(reservoir.W), sparse.csr_matrix(reservoir.Win)
