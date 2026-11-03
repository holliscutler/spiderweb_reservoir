"""Draw every figure into figures/ and write results/summary.md from results/results.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.stats import mannwhitneyu
from tasks import MEMORY
from topology import DEFAULT, LAYOUT, RESERVOIRS, SPOKES, UNITS, WEB, weights

HERE = Path(__file__).parent
RESULTS = json.loads((HERE / "results" / "results.json").read_text())
FIGURES = HERE / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {WEB: "#30659a", DEFAULT: "#c25b45"}
EXAMPLE_SEED = RESULTS["example"]["seed"]
# STIX is a Times-style journal font that ships with matplotlib, so figures look the same on every machine.
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 11, "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "axes.titlesize": 11, "legend.frameon": False})


def tidy(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def save(fig, name, note=None):
    if note:
        fig.text(0.5, -0.01, note, ha="center", va="top", color=MUTED, fontsize=9)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ci95(scores, seed=0):
    boot = np.random.default_rng(seed).choice(scores, size=(5000, len(scores))).mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


def draw_topology(example=1 + 6 * SPOKES):
    """Each reservoir in its natural drawing, one neuron's outgoing threads highlighted, and their matrices.

    The web is drawn on its rings and spokes. The default has no geometry, so it is drawn the standard way,
    neurons evenly spaced on a circle: on the web's layout its random connections would look like a web.
    """
    angles = 2 * np.pi * np.arange(UNITS) / UNITS
    layouts = {WEB: LAYOUT, DEFAULT: 10 * np.column_stack([np.cos(angles), np.sin(angles)])}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.6), gridspec_kw={"height_ratios": [1.1, 1]})
    for col, name in enumerate(RESERVOIRS):
        W, Win = weights(name, EXAMPLE_SEED)
        targets, sources = W.nonzero()
        xy = layouts[name]
        net = axes[0, col]
        net.add_collection(LineCollection(np.stack([xy[sources], xy[targets]], axis=1),
                                          colors=MUTED, linewidths=0.4, alpha=0.07))
        for b in targets[sources == example]:
            net.annotate("", xy=xy[b], xytext=xy[example], arrowprops=dict(
                arrowstyle="-|>", color=COLORS[name], lw=1.2, mutation_scale=9, shrinkA=3, shrinkB=3))
        inputs = np.unique(Win.nonzero()[0])
        net.scatter(*xy.T, s=10, color="#8a8984", zorder=2)
        net.scatter(*xy[inputs].T, s=34, color=COLORS[name], edgecolor=INK, linewidth=0.8, zorder=3,
                    label="Receives input")
        net.scatter(*xy[example], s=70, color=COLORS[name], edgecolor=INK, linewidth=1.2, marker="D", zorder=4,
                    label="Highlighted neuron")
        net.set_title(f"{name}: {W.nnz:,} connections, {len(inputs)} input neurons")
        net.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
        net.set_aspect("equal")
        net.axis("off")
        matrix = axes[1, col]
        matrix.spy(W, markersize=0.6, color=COLORS[name])
        matrix.set_title("Connection matrix")
        matrix.set_xlabel("From neuron")
        matrix.set_ylabel("To neuron")
        matrix.tick_params(length=0, labelsize=8)
    fig.tight_layout()
    save(fig, "1_topology.png")


def draw_forecast():
    example = RESULTS["example"]
    truth = np.array(example["truth"])
    steps = np.arange(len(truth))
    fig, (series, error) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    series.plot(steps, truth, color=INK, lw=2, label="True future value")
    for name in RESERVOIRS:
        pred = np.array(example[name])
        series.plot(steps, pred, color=COLORS[name], lw=1.5, label=f"{name} prediction")
        error.plot(steps, np.abs(pred - truth), color=COLORS[name], lw=1.2,
                   label=f"{name}: mean error {np.mean(np.abs(pred - truth)):.3f}")
    series.set_title(f"{example['task']} ahead", pad=30)
    series.set_ylabel("Value (standardized)")
    error.set_ylabel("Absolute error")
    error.set_xlabel("Test time step")
    for ax in (series, error):
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=3, fontsize=9)
        tidy(ax)
    save(fig, "2_forecast_example.png")


def task_stats(task):
    web, default = (np.array(RESULTS["tasks"][task][n]["test"]) for n in RESERVOIRS)
    higher = task == MEMORY
    better = (web.mean() - default.mean()) / default.mean() * (1 if higher else -1)
    return web, default, better, mannwhitneyu(web, default).pvalue


def draw_results():
    tasks = list(RESULTS["tasks"])
    fig, axes = plt.subplots(1, len(tasks), figsize=(16, 3.4), sharey=True)
    for ax, task in zip(axes, tasks):
        web, default, better, p = task_stats(task)
        for y, (name, scores) in enumerate(zip(RESERVOIRS, (web, default))):
            lo, hi = ci95(scores)
            ax.plot([lo, hi], [y, y], color=COLORS[name], lw=2.5, solid_capstyle="round", zorder=2)
            ax.scatter(scores, np.full(len(scores), y) + np.random.default_rng(y).uniform(-0.18, 0.18, len(scores)),
                       s=9, color=COLORS[name], alpha=0.35, linewidth=0, zorder=1)
            ax.scatter(scores.mean(), y, s=80, color=COLORS[name], edgecolor=SURFACE, linewidth=2, zorder=3)
        higher = task == MEMORY
        verdict = f"Web {abs(better):.0%} {'better' if better > 0 else 'worse'}, p = {p:.1g}"
        ax.set_title(f"{task}\n{'capacity, higher is better' if higher else 'error (NRMSE), lower is better'}\n"
                     f"{verdict}")
        ax.set_yticks(range(len(RESERVOIRS)), RESERVOIRS, color=INK)
        ax.set_ylim(len(RESERVOIRS) - 0.4, -0.6)
        ax.margins(x=0.12)
        tidy(ax, grid="x")
    save(fig, "3_results.png")


def hardware(name, key):
    return np.array([c[key] for c in RESULTS["hardware"][name]["per_seed"]])


def draw_hardware(task="Mackey-Glass, 100 steps"):
    fig, (hist, bars, scatter) = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.2, 1, 1.2]})
    bins = np.linspace(0, 21, 43)
    for name in RESERVOIRS:
        lengths = np.array(RESULTS["hardware"][name]["thread_lengths_example"])
        hist.hist(lengths, bins=bins, histtype="step", lw=2, color=COLORS[name],
                  label=f"{name}: median {np.median(lengths):.1f}")
    hist.set_title("Connection length")
    hist.set_xlabel("Wire length (ring spacing = 1)")
    hist.set_ylabel("Connections")
    hist.legend(fontsize=9)
    tidy(hist)

    x = np.arange(2)
    for i, name in enumerate(RESERVOIRS):
        values = [hardware(name, "wire_natural").mean(), hardware(name, "wire_optimized").mean()]
        rects = bars.bar(x + (i - 0.5) * 0.36, values, width=0.34, color=COLORS[name], label=name)
        bars.bar_label(rects, labels=[f"{v / 1000:.1f}k" for v in values], padding=3, fontsize=9, color=INK)
    bars.set_xticks(x, ["Web's layout", "Best placement found"])
    bars.set_title("Total wire per reservoir")
    bars.set_ylabel("Total wire length")
    bars.legend(fontsize=9, loc="upper right")
    tidy(bars)

    for name in RESERVOIRS:
        error = np.array(RESULTS["tasks"][task][name]["test"])
        scatter.scatter(hardware(name, "wire_optimized"), error, s=40, color=COLORS[name], edgecolor=SURFACE,
                        linewidth=1, label=name)
    scatter.set_title(f"Accuracy against wire: {task}")
    scatter.set_xlabel("Total wire length, best placement")
    scatter.set_ylabel("Error (NRMSE), lower is better")
    scatter.legend(fontsize=9)
    tidy(scatter, grid="both")
    save(fig, "4_hardware_cost.png")


def write_summary():
    lines = ["| Task | Spider web | Default random | Web vs default | p |", "|---|---|---|---|---|"]
    for task in RESULTS["tasks"]:
        web, default, better, p = task_stats(task)
        fmt = "{:.1f}" if task == MEMORY else "{:.4f}"
        cells = [f"{fmt.format(s.mean())} ({fmt.format(ci95(s)[0])}–{fmt.format(ci95(s)[1])})" for s in (web, default)]
        lines.append(f"| {task} | {cells[0]} | {cells[1]} | {abs(better):.0%} {'better' if better > 0 else 'worse'} "
                     f"| {p:.1g} |")
    lines += ["", "| Hardware (mean of 20) | Spider web | Default random | Web / default |", "|---|---|---|---|"]
    for key, label in [("ops_total", "Operations per time step"), ("wire_natural", "Total wire, web's layout"),
                       ("wire_optimized", "Total wire, best placement"), ("input_wire_natural", "Input wire")]:
        w, d = hardware(WEB, key).mean(), hardware(DEFAULT, key).mean()
        lines.append(f"| {label} | {w:,.0f} | {d:,.0f} | {w / d:.2f} |")
    lines += ["", "| Tuned settings | " + " | ".join(RESULTS["tasks"]) + " |",
              "|---|" + "---|" * len(RESULTS["tasks"])]
    for name in RESERVOIRS:
        cells = [", ".join(f"{v:g}" for v in RESULTS["tasks"][t][name]["params"].values()) for t in RESULTS["tasks"]]
        lines.append(f"| {name} (sr, lr, input scaling, ridge) | " + " | ".join(cells) + " |")
    (HERE / "results" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    draw_topology()
    draw_forecast()
    draw_results()
    draw_hardware()
    write_summary()
