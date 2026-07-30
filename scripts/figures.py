"""Aggregate all evaluated runs into the paper's figures, tables and numbers."""

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eodcomm.plotstyle import (  # noqa: E402
    SERIES, INK, INK2, MUTED, C_NULL, C_FULL, C_PRIVATE, C_CUE, C_SIGNAL,
    bars_ci, boot_ci, cohen_d, panel, perm_p, stars,
)

FIG = os.path.join(HERE, "..", "figs")
os.makedirs(FIG, exist_ok=True)
M = {}
NUM = {}


def load(pattern="results/metrics/*.json"):
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        M[d["name"]] = d
    print(f"loaded {len(M)} evaluated runs")


def group(prefix):
    """All runs whose name starts with prefix (before the _sN suffix)."""
    out = []
    for k, v in M.items():
        if re.match(rf"^{re.escape(prefix)}_s\d+$", k):
            out.append(v)
    return out


def vals(prefix, *path):
    out = []
    for r in group(prefix):
        x = r
        try:
            for p in path:
                x = x[p]
            if x is not None and np.isfinite(x):
                out.append(float(x))
        except (KeyError, TypeError):
            pass
    return np.array(out)


def num(k, v, fmt="{:.2f}"):
    NUM[k] = fmt.format(v) if isinstance(v, float) else str(v)
    return v


# ===========================================================================
# Figure 1 -- validation and throughput
# ===========================================================================
def fig1():
    val = json.load(open("results/physics_validation.json"))
    order = [
        ("monopole_field", "monopole field"),
        ("dipole_field", "dipole field"),
        ("field_with_wall_images", "field + wall images"),
        ("induced_dipole_moments", "induced moments"),
        ("clip_conductor_moments", "moment clipping"),
        ("morm_cd_baseline", "corollary-discharge baseline"),
        ("amp_intrinsic_baseline", "ampullary baseline"),
        ("mormyromast_pipeline", "mormyromast (self-image)"),
        ("mormyromast_collective_pipeline", "mormyromast (collective)"),
        ("knollen_pipeline", "knollenorgan"),
        ("ampullary_pipeline", "ampullary"),
    ]
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ks = [k for k, _ in order if k in val]
    labs = [l for k, l in order if k in val]
    v = np.array([max(val[k], 1e-17) for k in ks])
    y = np.arange(len(ks))
    cols = [C_SIGNAL if val[k] == 0 else C_FULL for k in ks]
    ax.barh(y, v, 0.62, color=cols, edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labs)
    ax.set_xscale("log"); ax.invert_yaxis()
    ax.set_xlabel("max relative deviation from reference NumPy")
    ax.axvline(1e-6, color=INK2, ls="--", lw=0.8, zorder=2)
    ax.text(1.3e-6, len(ks) - 0.4, "float32 epsilon", fontsize=6, color=INK2)
    for i, k in enumerate(ks):
        if val[k] == 0:
            ax.text(1.5e-17, i, " bit-exact", va="center", fontsize=6, color=INK)
    ax.set_xlim(1e-17, 1e-5)
    ax.grid(axis="x", alpha=0.8); ax.set_axisbelow(True)
    ax.set_title("Sensory pipeline reproduces the reference implementation", loc="left", pad=5)
    fig.savefig(f"{FIG}/fig1_validation.pdf"); plt.close(fig)
    num("ValMaxErr", f"$\\num{{{max(val.values()):.1e}}}$".replace("e-", "e{-}"))
    NUM["ValMaxErr"] = f"\\num{{{max(val.values()):.0e}}}"
    NUM["ValNChecks"] = str(len(val))


# ===========================================================================
# Figure 2 -- the three consequences, trained from scratch
# ===========================================================================
COND_A = [
    ("A_full", "intact", C_FULL),
    ("A_noknollen", "no detection", C_SIGNAL),
    ("A_noillum", "no illumination", C_CUE),
    ("A_private", "private probe", C_PRIVATE),
    ("A_noself", "no reafference", SERIES[6]),
    ("A_silent", "silent", MUTED),
]


def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    labs = [l for _, l, _ in COND_A]
    cols = [c for _, _, c in COND_A]

    g = [vals(p, "base", "eaten_per_ep") for p, _, _ in COND_A]
    panel(axes[0], "A", "Food per fish per episode")
    bars_ci(axes[0], labs, g, cols, "items", rot=35)
    full, sil = g[0], g[-1]
    num("FullEaten", float(np.mean(full)))
    num("SilentEaten", float(np.mean(sil)))
    num("ProbeGain", float(np.mean(full) / max(np.mean(sil), 1e-9)), "{:.1f}")
    num("ProbeD", cohen_d(full, sil), "{:.1f}")
    num("ProbeP", perm_p(full, sil), "{:.4f}")
    num("PrivateEaten", float(np.mean(g[3])))
    num("NoIllumEaten", float(np.mean(g[2])))
    num("NoKnollenEaten", float(np.mean(g[1])))
    num("NoSelfEaten", float(np.mean(g[4])))

    g2 = [vals(p, "base", "emit_rate") for p, _, _ in COND_A]
    panel(axes[1], "B", "Discharge probability")
    bars_ci(axes[1], labs, g2, cols, "P(emit)", rot=35)
    axes[1].axhline(0.5, color=INK2, ls=":", lw=0.8)
    num("FullEmit", float(np.mean(g2[0])), "{:.3f}")
    num("NoKnollenEmit", float(np.mean(g2[1])), "{:.3f}")

    g3 = [vals(p, "base", "nn_dist") for p, _, _ in COND_A]
    panel(axes[2], "C", "Nearest-neighbour distance")
    bars_ci(axes[2], labs, g3, cols, "cm", rot=35)
    num("FullNN", float(np.mean(g3[0])))
    num("NoKnollenNN", float(np.mean(g3[1])))
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_channels_scratch.pdf"); plt.close(fig)


# ===========================================================================
# Figure 3 -- decomposing the muting confound (frozen policy)
# ===========================================================================
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    runs = group("A_full")
    if not runs:
        plt.close(fig); return

    def comp(dv, key):
        out = []
        for r in runs:
            try:
                out.append(r["muting_decomposition"][dv][key]["mean"])
            except (KeyError, TypeError):
                pass
        return np.array(out)

    keys = [("total", "mute all three\n(upstream)", MUTED),
            ("private_share", "$-$reafference", C_PRIVATE),
            ("cue_share", "$-$illumination", C_CUE),
            ("signal_share", "$-$detection", C_SIGNAL)]
    for ax, dv, ttl, ylab, letter in [
        (axes[0], "eaten_target", "Muted fish's own intake", "$\\Delta$ items", "A"),
        (axes[1], "eaten_others", "Its neighbours' intake", "$\\Delta$ items", "B"),
        (axes[2], "nn_dist", "Group spacing", "$\\Delta$ cm", "C"),
    ]:
        panel(ax, letter, ttl)
        gs = [comp(dv, k) for k, _, _ in keys]
        bars_ci(ax, [l for _, l, _ in keys], gs, [c for _, _, c in keys], ylab, rot=30)
        ax.axhline(0, color=INK, lw=0.8)
        for i, gg in enumerate(gs):
            if len(gg) > 1:
                p = perm_p(gg, np.zeros_like(gg))
                m = np.mean(gg)
                ax.text(i, m + (0.06 * np.ptp(ax.get_ylim())) * np.sign(m or 1),
                        stars(p), ha="center", fontsize=6, color=INK)
        if dv == "eaten_target":
            num("MuteTargetTotal", float(np.mean(gs[0])))
            num("MuteTargetPriv", float(np.mean(gs[1])))
            num("MuteTargetCue", float(np.mean(gs[2])))
            num("MuteTargetSig", float(np.mean(gs[3])))
        if dv == "eaten_others":
            num("MuteOthersTotal", float(np.mean(gs[0])))
            num("MuteOthersPriv", float(np.mean(gs[1])))
            num("MuteOthersCue", float(np.mean(gs[2])))
            num("MuteOthersSig", float(np.mean(gs[3])))
            num("MuteOthersCueP", perm_p(gs[2], np.zeros_like(gs[2])), "{:.4f}")
            num("MuteOthersSigP", perm_p(gs[3], np.zeros_like(gs[3])), "{:.4f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_muting_decomposition.pdf"); plt.close(fig)


# ===========================================================================
# Figure 4 -- what the pulse carries
# ===========================================================================
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    tgts = [("food", "food"), ("will_eat", "about to eat"), ("dominance", "dominance"),
            ("movement", "movement"), ("danger", "danger")]

    panel(axes[0], "A", "Positive signalling (cost 0)")
    gs = [vals("A_full", "positive_signaling", t, "norm") for t, _ in tgts if t != "will_eat"]
    ls = [l for t, l in tgts if t != "will_eat"]
    bars_ci(axes[0], ls, gs, SERIES, "$I(m;S)/H(S)$", rot=25)
    if len(gs[0]):
        num("PSFood", float(np.mean(gs[0])), "{:.3f}")

    panel(axes[1], "B", "Receiver-decodable content")
    gs2 = [vals("A_full", "content", t, "delta") for t, _ in tgts]
    bars_ci(axes[1], [l for _, l in tgts], gs2, SERIES, "$\\Delta R^2$ / $\\Delta$AUC", rot=25)
    axes[1].axhline(0, color=INK, lw=0.8)
    if len(gs2[0]):
        num("ContentFood", float(np.mean(gs2[0])), "{:.3f}")
        num("ContentFoodP", perm_p(gs2[0], np.zeros_like(gs2[0])), "{:.4f}")
    if len(gs2[2]):
        num("ContentDom", float(np.mean(gs2[2])), "{:.3f}")

    panel(axes[2], "C", "Discharge rate is state-dependent")
    a = vals("A_full", "base", "emit_rate_nofood")
    b = vals("A_full", "base", "emit_rate_food")
    bars_ci(axes[2], ["no food\nwithin 10\\,cm", "food\nwithin 10\\,cm"], [a, b],
            [MUTED, C_FULL], "P(emit)")
    if len(a) and len(b):
        num("EmitNoFood", float(np.mean(a)), "{:.3f}")
        num("EmitFood", float(np.mean(b)), "{:.3f}")
        num("EmitFoodP", perm_p(a, b), "{:.4f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_content.pdf"); plt.close(fig)


# ===========================================================================
# Figure 5 -- positive listening under counterfactual channels
# ===========================================================================
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    runs = group("A_full")
    if not runs:
        plt.close(fig); return

    panel(axes[0], "A", "Interventional causal influence")
    ci = np.array([r["cie"]["mean"] for r in runs if "cie" in r])
    cn = np.array([r["cie"]["null"] for r in runs if "cie" in r])
    bars_ci(axes[0], ["$\\dosim(e_j)$\ncontrast", "noise\nfloor"], [ci, cn], [C_FULL, C_NULL],
            "$D_{\\mathrm{KL}}$ (nats)")
    axes[0].set_yscale("log")
    if len(ci):
        num("CIE", float(np.mean(ci)), "{:.4f}")
        num("CIENull", float(np.mean(cn)), "{:.5f}")
        num("CIERatio", float(np.mean(ci) / max(np.mean(cn), 1e-12)), "{:.0f}")

    panel(axes[1], "B", "Positive listening")
    pl = np.array([r["positive_listening"]["shift_null"] for r in runs if "positive_listening" in r])
    pz = np.array([r["positive_listening"]["zero_null"] for r in runs if "positive_listening" in r])
    bars_ci(axes[1], ["vs time-shift\n(rate-matched)", "vs silence\n(out of distribution)"],
            [pl, pz], [C_FULL, C_NULL], "$L^1$ policy divergence")
    if len(pl):
        num("PL", float(np.mean(pl)), "{:.3f}")

    panel(axes[2], "C", "Phantom pulses: receiver response")
    rates, means, los, his = [], [], [], []
    for key in sorted({k for r in runs for k in r.get("phantom_dose", {})},
                      key=lambda s: float(s.split("_")[1])):
        v = np.array([r["phantom_dose"][key]["mean"] for r in runs if key in r.get("phantom_dose", {})])
        if not len(v):
            continue
        m, lo, hi = boot_ci(v)
        rates.append(float(key.split("_")[1])); means.append(m); los.append(lo); his.append(hi)
    if rates:
        axes[2].fill_between(rates, los, his, color=C_SIGNAL, alpha=0.18, lw=0)
        axes[2].plot(rates, means, "o-", color=C_SIGNAL, ms=4, mfc="white", mew=1.2)
        axes[2].axhline(0, color=INK, lw=0.8)
        axes[2].set_xlabel("phantom discharge rate"); axes[2].set_ylabel("$\\Delta$ neighbours' intake")
        num("PhantomMax", float(np.max(np.abs(means))), "{:.3f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_listening.pdf"); plt.close(fig)


# ===========================================================================
# Figure 6 -- cost converts a cue into a signal (SSI)
# ===========================================================================
def fig6():
    import torch
    from eodcomm.train import load_agent
    from eodcomm.metrics import sender_shaping_index, collect

    pairs = [(0.0, "A_full", "A_noknollen"),
             (0.02, "B_hear_c0.02", "B_deaf_c0.02"),
             (0.06, "B_hear_c0.06", "B_deaf_c0.06")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    costs, ssis, emit_h, emit_d, psf = [], [], [], [], []
    ctx = None
    for c, hp, dp in pairs:
        hs = sorted(glob.glob(f"results/runs/{hp}_s*.pt"))
        ds = sorted(glob.glob(f"results/runs/{dp}_s*.pt"))
        if not hs or not ds:
            continue
        if ctx is None:
            tr, env, _ = load_agent(hs[0], env_overrides=dict(batch=64))
            rec = collect(tr, env, 96, seed=99)
            ctx = rec["obs"].reshape(-1, env.obs_dim)
            idx = torch.randperm(ctx.shape[0], device=ctx.device)[:4096]
            ctx = ctx[idx]
            del tr, env
        nh = [load_agent(p, env_overrides=dict(batch=2))[0].net for p in hs]
        nd = [load_agent(p, env_overrides=dict(batch=2))[0].net for p in ds]
        r = sender_shaping_index(nh, nd, ctx)
        costs.append(c); ssis.append(r["ssi"])
        emit_h.append(np.mean(vals(hp, "base", "emit_rate")))
        emit_d.append(np.mean(vals(dp, "base", "emit_rate")))
        psf.append(np.mean(vals(hp, "positive_signaling", "food", "norm")))
        torch.cuda.empty_cache()

    if costs:
        panel(axes[0], "A", "Sender Shaping Index")
        axes[0].plot(costs, ssis, "o-", color=C_SIGNAL, ms=5, mfc="white", mew=1.4)
        axes[0].axhline(0, color=INK, lw=0.9)
        axes[0].set_xlabel("metabolic cost per discharge $\\lambda$")
        axes[0].set_ylabel("SSI")
        axes[0].text(0.02, 0.92, "signal", transform=axes[0].transAxes, fontsize=6.4, color=INK2)
        axes[0].text(0.02, 0.06, "cue", transform=axes[0].transAxes, fontsize=6.4, color=INK2)
        num("SSIZero", ssis[0], "{:.3f}")
        num("SSIMax", max(ssis), "{:.3f}")
        num("SSICostMax", costs[int(np.argmax(ssis))], "{:.2f}")

        panel(axes[1], "B", "Discharge rate")
        axes[1].plot(costs, emit_h, "o-", color=C_FULL, ms=5, mfc="white", mew=1.4, label="receivers hear")
        axes[1].plot(costs, emit_d, "s--", color=MUTED, ms=4.4, mfc="white", mew=1.2, label="receivers deaf")
        axes[1].set_xlabel("metabolic cost $\\lambda$"); axes[1].set_ylabel("P(emit)")
        axes[1].legend(loc="best")

        panel(axes[2], "C", "Positive signalling about food")
        axes[2].plot(costs, psf, "o-", color=C_CUE, ms=5, mfc="white", mew=1.4)
        axes[2].set_xlabel("metabolic cost $\\lambda$"); axes[2].set_ylabel("$I(m;\\text{food})/H$")
        if len(psf) > 1:
            num("PSFoodHighCost", psf[-1], "{:.3f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig6_ssi.pdf"); plt.close(fig)


# ===========================================================================
# Figure 7 -- the phase diagram
# ===========================================================================
COSTS = [0.0, 0.01, 0.02, 0.04, 0.08]
PREDS = [0.0, 5.0, 20.0]


def grid_of(field, econ, *path):
    g = np.full((len(PREDS), len(COSTS)), np.nan)
    for i, p in enumerate(PREDS):
        for j, c in enumerate(COSTS):
            v = vals(f"C_c{c}_p{p}_{econ}", *path)
            if len(v):
                g[i, j] = np.mean(v)
    return g


def fig7():
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.9))
    specs = [
        ("emit_rate", "Discharge probability", "P(emit)", "viridis"),
        ("silence_index", "Submissive silence", "$1-r_{\\rm sub}/r_{\\rm dom}$", "cividis"),
        ("struck_per_ep", "Predation mortality", "strikes / episode", "magma"),
        ("eaten_per_ep", "Food per fish", "items", "viridis"),
    ]
    for row, econ, ename in [(0, "cmp", "competition"), (1, "coop", "cooperation")]:
        for col, (f, ttl, cb, cmap) in enumerate(specs):
            ax = axes[row, col]
            g = grid_of(f, econ, "base", f)
            im = ax.imshow(g, cmap=cmap, aspect="auto", origin="lower")
            ax.set_xticks(range(len(COSTS)))
            ax.set_xticklabels([f"{c:g}" for c in COSTS], fontsize=6)
            ax.set_yticks(range(len(PREDS)))
            ax.set_yticklabels([f"{p:g}" for p in PREDS], fontsize=6)
            if row == 1:
                ax.set_xlabel("cost $\\lambda$", fontsize=6.8)
            if col == 0:
                ax.set_ylabel(f"{ename}\npredation $\\rho$", fontsize=6.8)
            if row == 0:
                ax.set_title(ttl, loc="left", fontsize=7.2, pad=3)
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    if np.isfinite(g[i, j]):
                        ax.text(j, i, f"{g[i,j]:.2f}", ha="center", va="center",
                                fontsize=5.2, color="white" if g[i, j] < np.nanmean(g) else "black")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=5.4)
            cbar.set_label(cb, fontsize=5.6)
            ax.grid(False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig7_phase_grid.pdf"); plt.close(fig)

    e0 = grid_of("emit_rate", "cmp", "base", "emit_rate")
    if np.isfinite(e0).any():
        num("EmitCostZero", float(e0[0, 0]), "{:.3f}")
        num("EmitCostHigh", float(e0[0, -1]), "{:.3f}")
        num("EmitPredHigh", float(e0[-1, 0]), "{:.3f}")
        num("EmitBoth", float(e0[-1, -1]), "{:.3f}")


# ===========================================================================
# Figure 8 -- regime classification
# ===========================================================================
REGIMES = [
    ("private active sensing", SERIES[0]),
    ("public-information signalling", SERIES[2]),
    ("cooperative communication", SERIES[3]),
    ("competitive / deceptive", SERIES[1]),
    ("submissive silence", SERIES[4]),
    ("cryptic low-rate sensing", SERIES[6]),
]


def classify(cell):
    """Rule-based regime label from the metric vector of one grid cell.

    Thresholds are set from the pooled distribution of the cost-0, predation-0
    cells, which is the reference condition, rather than by hand.
    """
    r, sil, mort, cue, sig, coop = cell
    if not np.isfinite(r):
        return -1
    if r < TH["rate_low"] and mort > TH["mort_hi"]:
        return 5           # cryptic low-rate sensing under predation
    if sil > TH["sil_hi"]:
        return 4           # submissive silence
    if r < TH["rate_low"]:
        return 5
    if sig > TH["sig_hi"] and coop:
        return 2           # cooperative communication
    if sig > TH["sig_hi"]:
        return 3           # competitive / deceptive use of the channel
    if cue > TH["cue_hi"]:
        return 1           # public-information signalling (cue exploitation)
    return 0               # private active sensing


TH = {}


def fig8():
    ref_r = grid_of("emit_rate", "cmp", "base", "emit_rate")
    if not np.isfinite(ref_r).any():
        return
    allr = np.concatenate([grid_of("emit_rate", e, "base", "emit_rate").ravel()
                           for e in ("cmp", "coop")])
    alls = np.concatenate([grid_of("silence_index", e, "base", "silence_index").ravel()
                           for e in ("cmp", "coop")])
    allm = np.concatenate([grid_of("struck_per_ep", e, "base", "struck_per_ep").ravel()
                           for e in ("cmp", "coop")])
    TH["rate_low"] = float(np.nanpercentile(allr, 30))
    TH["sil_hi"] = float(np.nanpercentile(alls, 75))
    TH["mort_hi"] = float(np.nanpercentile(allm[allm > 0], 40)) if (allm > 0).any() else 1e9
    TH["sig_hi"] = float(np.nanpercentile(
        np.concatenate([grid_of("x", e, "content", "food", "delta").ravel() for e in ("cmp", "coop")]), 70))
    TH["cue_hi"] = TH["sig_hi"] * 0.5

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, econ, ename in [(axes[0], "cmp", "competition"), (axes[1], "coop", "cooperation")]:
        r = grid_of("emit_rate", econ, "base", "emit_rate")
        sil = grid_of("silence_index", econ, "base", "silence_index")
        mort = grid_of("struck_per_ep", econ, "base", "struck_per_ep")
        cnt = grid_of("x", econ, "content", "food", "delta")
        lab = np.full(r.shape, -1)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                lab[i, j] = classify((r[i, j], sil[i, j], mort[i, j], cnt[i, j],
                                      cnt[i, j], econ == "coop"))
        cmap = plt.matplotlib.colors.ListedColormap([c for _, c in REGIMES])
        ax.imshow(np.ma.masked_less(lab, 0), cmap=cmap, vmin=0, vmax=len(REGIMES) - 1,
                  aspect="auto", origin="lower")
        ax.set_xticks(range(len(COSTS))); ax.set_xticklabels([f"{c:g}" for c in COSTS], fontsize=6)
        ax.set_yticks(range(len(PREDS))); ax.set_yticklabels([f"{p:g}" for p in PREDS], fontsize=6)
        ax.set_xlabel("metabolic cost $\\lambda$", fontsize=6.8)
        ax.set_ylabel("predation $\\rho$", fontsize=6.8)
        ax.set_title(ename, loc="left", fontsize=7.4, pad=3)
        ax.grid(False)
        for i in range(lab.shape[0]):
            for j in range(lab.shape[1]):
                if lab[i, j] >= 0:
                    ax.text(j, i, str(lab[i, j] + 1), ha="center", va="center",
                            fontsize=6, color="white", fontweight="bold")
    handles = [plt.Line2D([], [], marker="s", ls="none", ms=6, color=c,
                          label=f"{i+1}. {n}") for i, (n, c) in enumerate(REGIMES)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.2,
               bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig8_regimes.pdf"); plt.close(fig)


# ===========================================================================
def write_numbers():
    NUM["NumRuns"] = str(len(glob.glob("results/runs/*.pt")))
    NUM["NumEval"] = str(len(M))
    with open("paper/numbers.tex", "w") as f:
        f.write("% auto-generated by scripts/figures.py -- do not edit\n")
        for k, v in sorted(NUM.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote paper/numbers.tex with {len(NUM)} macros")


if __name__ == "__main__":
    load()
    for fn in (fig1, fig2, fig3, fig4, fig5, fig7, fig8, fig6):
        try:
            fn()
            print("ok", fn.__name__, flush=True)
        except Exception as e:
            import traceback
            print(f"FAIL {fn.__name__}: {e}\n{traceback.format_exc()}", flush=True)
    write_numbers()
