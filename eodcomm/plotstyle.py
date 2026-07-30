"""Shared figure style.

Categorical hues are a fixed-order, CVD-validated set (adjacent-pair deltaE >= 9.1
under protanopia, >= 19.6 for normal vision).  Series are always direct-labelled
or legended in text ink, so identity never rests on colour alone.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e2e1dc"

# semantic assignments used consistently across every figure
C_FULL = SERIES[0]      # intact channel
C_PRIVATE = SERIES[1]   # private probe / reafference
C_CUE = SERIES[2]       # shared illumination (exploitable cue)
C_SIGNAL = SERIES[4]    # knollen detection (candidate signal)
C_SILENT = MUTED
C_NULL = "#b9b8b2"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 7.2,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.4,
    "axes.titleweight": "bold",
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 6.6,
    "legend.frameon": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def panel(ax, letter, title=None):
    ax.text(-0.16, 1.10, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left", color=INK)
    if title:
        ax.set_title(title, loc="left", pad=4)
    ax.grid(axis="y", alpha=0.85)
    ax.set_axisbelow(True)
    return ax


def boot_ci(x, n=5000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], x[0], x[0]
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def bars_ci(ax, labels, groups, colors, ylabel="", show_pts=True, rot=0):
    """Bar + bootstrap CI + individual seed points."""
    xs = np.arange(len(labels))
    for i, (lab, g) in enumerate(zip(labels, groups)):
        m, lo, hi = boot_ci(g)
        ax.bar(i, m, 0.62, color=colors[i % len(colors)], zorder=2,
               edgecolor="white", linewidth=0.8)
        ax.plot([i, i], [lo, hi], color=INK, lw=1.1, zorder=4, solid_capstyle="butt")
        if show_pts and len(g) > 1:
            j = (np.random.default_rng(i).random(len(g)) - 0.5) * 0.26
            ax.plot(i + j, g, "o", ms=2.4, mfc="white", mec=INK2, mew=0.6, zorder=5, ls="none")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center")
    ax.set_ylabel(ylabel)
    return ax


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if s == 0:
        return np.nan
    d = (a.mean() - b.mean()) / s
    # Hedges' g small-sample correction
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def perm_p(a, b, n=20000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n + 1)


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
