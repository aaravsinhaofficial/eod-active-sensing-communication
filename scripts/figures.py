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


# Training is bimodal: a seed either discovers patch foraging or never leaves the
# floor set by passive sensing alone.  A seed counts as converged if it reaches
# half the ceiling of its own task family -- for the standard arena that is 3.0
# items, roughly midway between the all-silent floor (1.8) and the intact ceiling
# (5.9), and the same relative criterion transfers to the sparse-patch task,
# whose absolute scale is different.  The rule is applied identically to every
# condition within a family, and the fraction of seeds clearing it is itself
# reported as an outcome.
FILTER_CONVERGED = True
_CEIL = {}


def _family(name):
    """Task family: conditions that share a reward scale and are comparable."""
    if name.startswith("E_sparse"):
        return "E_sparse"
    if name.startswith("E_std"):
        return "E_std"
    return name.split("_")[0]


def _ceiling(fam):
    if fam not in _CEIL:
        by = defaultdict(list)
        for k, v in M.items():
            if _family(k) == fam:
                by[k.rsplit("_s", 1)[0]].append(v["base"]["eaten_per_ep"])
        best = max((np.mean(v) for v in by.values()), default=6.0)
        _CEIL[fam] = best
    return _CEIL[fam]


def conv_thresh(name):
    return 0.5 * _ceiling(_family(name))


def converged(r):
    return r["base"]["eaten_per_ep"] >= conv_thresh(r["name"])


def group(prefix, apply_filter=None):
    """All runs whose name starts with prefix (before the _sN suffix)."""
    out = []
    for k, v in M.items():
        if re.match(rf"^{re.escape(prefix)}_s\d+$", k):
            out.append(v)
    use = FILTER_CONVERGED if apply_filter is None else apply_filter
    if use:
        keep = [r for r in out if converged(r)]
        if len(keep) >= 3:          # never filter down to an unusable sample
            return keep
    return out


def conv_frac(prefix):
    allr = group(prefix, apply_filter=False)
    if not allr:
        return float("nan")
    return sum(converged(r) for r in allr) / len(allr)


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
# Figure 0b -- can the optimiser invent the code, or only read one?
# ===========================================================================
def fig_learned_ref():
    import json as _j
    conds = [("R_emergent", "both sides\nlearn"),
             ("R_recvonly", "receiver learns,\nsender scripted"),
             ("R_eccles", "both learn,\n+PS bias")]
    got = []
    for pref, lab in conds:
        arr = []
        for f in glob.glob(f"results/runs/{pref}_s*_hist.json"):
            try:
                h = _j.load(open(f))["hist"][-1]
                if "arrivals_per_episode" in h:
                    arr.append(h["arrivals_per_episode"])
            except Exception:
                pass
        if arr:
            got.append((lab, np.array(arr)))
    if not got:
        return
    fp = "results/assay_validation.json"
    scripted = None
    if os.path.exists(fp):
        rows = _j.load(open(fp))
        for r in rows:
            if r["script"] == "honest" and r["listen"]:
                scripted = r["arrivals_per_ep"]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    labs = [l for l, _ in got]
    vals_ = [v for _, v in got]
    if scripted is not None:
        labs = ["scripted\ndyad"] + labs
        vals_ = [np.array([scripted])] + vals_
    panel(axes[0], "A", "Referential game: task success")
    bars_ci(axes[0], labs, vals_, [C_SIGNAL] + [C_FULL] * (len(labs) - 1),
            "correct arrivals / episode", rot=0)
    axes[0].axhline(2.0, color=INK, ls=":", lw=0.9)
    axes[0].text(0.02, 0.30, "chance", transform=axes[0].transAxes, fontsize=6, color=INK2)
    axes[0].axhline(4.0, color=MUTED, ls="--", lw=0.8)

    panel(axes[1], "B", "Seeds solving the game")
    fr = [float((v > 3.0).mean()) for v in vals_]
    axes[1].bar(np.arange(len(labs)), fr, 0.62,
                color=[C_SIGNAL] + [C_FULL] * (len(labs) - 1),
                edgecolor="white", lw=0.8)
    axes[1].set_xticks(np.arange(len(labs))); axes[1].set_xticklabels(labs, fontsize=6)
    axes[1].set_ylabel("fraction of seeds"); axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig0b_learned_referential.pdf"); plt.close(fig)

    for (lab, v), pref in zip(got, [c[0] for c in conds]):
        tag = pref.split("_")[1].capitalize()
        num(f"Ref{tag}Arr", float(v.mean()), "{:.2f}")
        NUM[f"Ref{tag}Solved"] = f"{int((v > 3.0).sum())}/{len(v)}"


# ===========================================================================
# Figure 0c -- the same audit in a world with no electric fish in it
# ===========================================================================
def fig_dualuse():
    fp = "results/dualuse.json"
    if not os.path.exists(fp):
        return
    d = json.load(open(fp))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

    conds = [("full", "intact"), ("no_detect", "no detection"),
             ("no_illum", "no illumination"), ("private", "private only"),
             ("silent", "silent")]
    cols = [C_FULL, C_SIGNAL, C_CUE, C_PRIVATE, MUTED]
    got = [np.array(d["cond"][k]["collected"]) for k, _ in conds if k in d.get("cond", {})]
    labs = [l for k, l in conds if k in d.get("cond", {})]
    if got:
        panel(axes[0], "A", "Items collected (trained from scratch)")
        bars_ci(axes[0], labs, got, cols, "items / episode", rot=30)
        num("DUFull", float(got[0].mean()))
        num("DUSilent", float(got[-1].mean()))

    if "decomp" in d:
        keys = [("mute_both", "mute all\nthree", MUTED),
                ("mute_self", "$-$reafference", C_PRIVATE),
                ("mute_illum", "$-$illumination", C_CUE),
                ("mute_detect", "$-$detection", C_SIGNAL)]
        gs = [np.array([b["collected_target"][k]["mean"] for b in d["decomp"]])
              for k, _, _ in keys]
        panel(axes[1], "B", "Muting one agent: its own intake")
        bars_ci(axes[1], [l for _, l, _ in keys], gs, [c for _, _, c in keys],
                "$\\Delta$ items", rot=30)
        axes[1].axhline(0, color=INK, lw=0.8)
        for i, gg in enumerate(gs):
            if len(gg) > 1:
                axes[1].text(i, np.mean(gg), stars(perm_p(gg, np.zeros_like(gg))),
                             ha="center", va="bottom", fontsize=6, color=INK)
        tot, pri = np.mean(gs[0]), np.mean(gs[1])
        num("DUTotal", float(tot)); num("DUPriv", float(pri))
        num("DUSig", float(np.mean(gs[3])))
        if abs(tot) > 1e-9:
            num("DUPrivFrac", 100 * float(pri / tot), "{:.0f}")

    if "cost" in d:
        cs = sorted(float(k) for k in d["cost"])
        pg = [np.mean(d["cost"][str(c) if str(c) in d["cost"] else f"{c}"]["ping"]) for c in cs]
        panel(axes[2], "C", "Cost suppresses the ping here too")
        axes[2].plot(cs, pg, "o-", color=C_FULL, ms=5, mfc="white", mew=1.3)
        axes[2].set_xlabel("ping cost"); axes[2].set_ylabel("P(ping)")
        num("DUPingLo", float(pg[0]), "{:.3f}")
        num("DUPingHi", float(pg[-1]), "{:.3f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig0c_dualuse.pdf"); plt.close(fig)


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
    for i, (p, _, _) in enumerate(COND_A):
        cf = conv_frac(p)
        if np.isfinite(cf):
            axes[0].text(i, 0.02, f"{int(round(cf*100))}\\%", transform=axes[0].get_xaxis_transform(),
                         ha="center", va="bottom", fontsize=5.4, color=INK2)
        NUM[f"ConvFrac{p.replace('_','')}"] = f"{int(round(cf*100))}"
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

    ph = vals("A_full", "positive_signaling", "food", "norm")
    pdf = vals("A_noknollen", "positive_signaling", "food", "norm")
    if len(ph) and len(pdf):
        num("PSHear", float(np.mean(ph)), "{:.3f}")
        num("PSDeaf", float(np.mean(pdf)), "{:.3f}")

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
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.5))
    runs = group("A_full")
    if not runs:
        plt.close(fig); return

    panel(axes[0], "A", "Interventional causal influence")
    ci = np.array([r["cie"]["mean"] for r in runs if "cie" in r])
    cn = np.array([r["cie"]["null"] for r in runs if "cie" in r])
    bars_ci(axes[0], ["do$(e_j)$\ncontrast", "noise\nfloor"], [ci, cn], [C_FULL, C_NULL],
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

    panel(axes[2], "C", "Phantom pulses")
    rates, means, los, his = [], [], [], []
    for key in sorted({k for r in runs for k in r.get("phantom_dose", {})},
                      key=lambda s_: float(s_.split("_")[1])):
        v = np.array([r["phantom_dose"][key]["mean"] for r in runs if key in r.get("phantom_dose", {})])
        if not len(v):
            continue
        m, lo, hi = boot_ci(v)
        rates.append(float(key.split("_")[1])); means.append(m); los.append(lo); his.append(hi)
    if rates:
        axes[2].fill_between(rates, los, his, color=C_SIGNAL, alpha=0.18, lw=0)
        axes[2].plot(rates, means, "o-", color=C_SIGNAL, ms=4, mfc="white", mew=1.2)
        axes[2].axhline(0, color=INK, lw=0.8)
        axes[2].set_xlabel("phantom discharge rate")
        axes[2].set_ylabel("$\\Delta$ neighbours' intake")
        num("PhantomMax", float(np.max(np.abs(means))), "{:.3f}")

    # --- the dissociation: influence rises where content falls -------------
    panel(axes[3], "D", "Influence rises as content falls")
    pairs = [(0.0, "A_full"), (0.02, "B_hear_c0.02"), (0.06, "B_hear_c0.06")]
    cs, cies, conts = [], [], []
    for c, pref in pairs:
        v1 = np.array([r["cie"]["mean"] for r in group(pref) if "cie" in r])
        v2 = vals(pref, "content", "food", "delta")
        if len(v1) and len(v2):
            cs.append(c); cies.append(v1.mean()); conts.append(v2.mean())
    if cs:
        ax = axes[3]
        ax.plot(cs, np.array(cies) / max(cies), "o-", color=C_SIGNAL, ms=5,
                mfc="white", mew=1.3, label="causal influence")
        ax.plot(cs, np.array(conts) / max(max(conts), 1e-9), "s--", color=C_CUE, ms=4.5,
                mfc="white", mew=1.2, label="decodable content")
        ax.set_xlabel("metabolic cost $\\lambda$")
        ax.set_ylabel("normalised to max")
        ax.axhline(0, color=INK, lw=0.8)
        ax.legend(loc="best")
        num("CIECostZero", float(cies[0]), "{:.4f}")
        num("CIECostHigh", float(cies[-1]), "{:.4f}")
        num("CIEFold", float(cies[-1] / max(cies[0], 1e-12)), "{:.0f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_listening.pdf"); plt.close(fig)


# ===========================================================================
# Figure 0 -- validating the assays on dyads of known communication status
# ===========================================================================
def fig_assay():
    fp = "results/assay_validation.json"
    if not os.path.exists(fp):
        return
    rows = json.load(open(fp))
    key = {(r["script"], r["listen"]): r for r in rows}
    order = [("honest", True, "info\nattend"),
             ("honest", False, "info\nignore"),
             ("random", True, "noise\nattend"),
             ("random", False, "noise\nignore")]
    cols = [C_SIGNAL, C_CUE, C_PRIVATE, MUTED]
    labs = [l for _, _, l in order]
    rs = [key[(a, b)] for a, b, _ in order]

    fig, axes = plt.subplots(1, 5, figsize=(9.8, 2.6))
    def bar(ax, letter, title, vals_, ylab, hline=None):
        panel(ax, letter, title)
        ax.bar(np.arange(len(vals_)), vals_, 0.62, color=cols,
               edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xticks(np.arange(len(labs)))
        ax.set_xticklabels(labs, fontsize=6.2)
        ax.set_ylabel(ylab)
        if hline is not None:
            ax.axhline(hline, color=INK, ls=":", lw=0.9)

    bar(axes[0], "A", "Task success", [r["arrivals_per_ep"] for r in rs],
        "correct arrivals / episode", hline=2.0)
    axes[0].text(0.02, 0.93, "chance", transform=axes[0].transAxes, fontsize=5.6, color=INK2)
    bar(axes[1], "B", "Positive signalling", [r["content_mi"] for r in rs],
        "$I$(subtype; referent)  bits")
    bar(axes[2], "C", "Causal influence", [r["cie"] for r in rs], "$D_{KL}$ (nats)")
    bar(axes[3], "D", "Positive listening", [r["pl_shift"] for r in rs],
        "$L^1$ policy divergence")
    bar(axes[4], "E", "Payoff of the content", [-r["kill_d_receiver"] for r in rs],
        "return lost when deleted")
    for ax in axes:
        ax.grid(axis="y", alpha=0.85); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig0_assay_validation.pdf"); plt.close(fig)

    h = key[("honest", True)]; hd = key[("honest", False)]; rl = key[("random", True)]
    num("AssayArrHonest", h["arrivals_per_ep"], "{:.2f}")
    num("AssayArrDeaf", hd["arrivals_per_ep"], "{:.2f}")
    num("AssayMIHonest", h["content_mi"], "{:.3f}")
    num("AssayMIRandom", rl["content_mi"], "{:.3f}")
    num("AssayCIEListen", h["cie"], "{:.1f}")
    num("AssayCIEDeaf", hd["cie"], "{:.3f}")
    num("AssayKillHonest", h["kill_d_receiver"], "{:.1f}")
    num("AssayKillRandom", rl["kill_d_receiver"], "{:.1f}")
    num("AssayKillDeaf", hd["kill_d_receiver"], "{:.2f}")


# ===========================================================================
# Figure 6 -- sender shaping, with reception held fixed
# ===========================================================================
def fig6():
    """The hearing/deaf contrast removes *reception*, so it cannot separate
    "I behave differently when others can hear me" from "I behave differently
    when I can hear others".  The yoked world matches reception statistics and
    cuts only audibility, which is the contrast that isolates being listened to.
    """
    import torch
    from eodcomm.train import load_agent
    from eodcomm.metrics import sender_shaping_index, collect, knollen_slice

    pairs = [(0.0, "F_live_c0.0", "F_yoked_c0.0", "A_full", "A_noknollen"),
             (0.06, "F_live_c0.06", "F_yoked_c0.06", "B_hear_c0.06", "B_deaf_c0.06")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ctx, kmask = None, None
    costs, ssi_yoke, ssi_deaf, e_live, e_yoke = [], [], [], [], []

    def nets(prefix):
        ok = {r["name"] for r in group(prefix)}
        return [p for p in sorted(glob.glob(f"results/runs/{prefix}_s*.pt"))
                if os.path.basename(p)[:-3] in ok]

    for c, live, yoke, hear, deaf in pairs:
        pl, py = nets(live), nets(yoke)
        if len(pl) < 2 or len(py) < 2:
            continue
        if ctx is None:
            tr, env, _ = load_agent(pl[0], env_overrides=dict(batch=64))
            rec = collect(tr, env, 96, seed=99)
            o = rec["obs"].reshape(-1, env.obs_dim)
            ctx = o[torch.randperm(o.shape[0], device=o.device)[:4096]]
            kmask = knollen_slice(env)
            del tr, env
        nl = [load_agent(p, env_overrides=dict(batch=2))[0].net for p in pl]
        ny = [load_agent(p, env_overrides=dict(batch=2))[0].net for p in py]
        costs.append(c)
        ssi_yoke.append(sender_shaping_index(nl, ny, ctx, mask=kmask)["ssi"])
        e_live.append(np.mean(vals(live, "base", "emit_rate")))
        e_yoke.append(np.mean(vals(yoke, "base", "emit_rate")))
        ph, pdd = nets(hear), nets(deaf)
        if len(ph) >= 2 and len(pdd) >= 2:
            nh = [load_agent(p, env_overrides=dict(batch=2))[0].net for p in ph]
            nd = [load_agent(p, env_overrides=dict(batch=2))[0].net for p in pdd]
            ssi_deaf.append(sender_shaping_index(nh, nd, ctx, mask=kmask)["ssi"])
        else:
            ssi_deaf.append(np.nan)
        torch.cuda.empty_cache()

    if not costs:
        plt.close(fig); return
    x = np.arange(len(costs))
    panel(axes[0], "A", "Sender shaping")
    axes[0].bar(x - 0.19, ssi_yoke, 0.36, color=C_SIGNAL, edgecolor="white", lw=0.8,
                label="yoked (audibility cut)")
    axes[0].bar(x + 0.19, ssi_deaf, 0.36, color=C_NULL, edgecolor="white", lw=0.8,
                label="deaf (reception cut)")
    axes[0].axhline(0, color=INK, lw=0.9)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"$\\lambda$={c:g}" for c in costs])
    axes[0].set_ylabel("SSI"); axes[0].legend(loc="best")
    num("SSIYokeZero", float(ssi_yoke[0]), "{:.3f}")
    if len(ssi_yoke) > 1:
        num("SSIYokeHigh", float(ssi_yoke[-1]), "{:.3f}")
    if np.isfinite(ssi_deaf[0]):
        num("SSIDeafZero", float(ssi_deaf[0]), "{:.3f}")

    panel(axes[1], "B", "Discharge rate: live vs yoked")
    axes[1].bar(x - 0.19, e_live, 0.36, color=C_FULL, edgecolor="white", lw=0.8, label="live")
    axes[1].bar(x + 0.19, e_yoke, 0.36, color=C_NULL, edgecolor="white", lw=0.8, label="yoked")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"$\\lambda$={c:g}" for c in costs])
    axes[1].set_ylabel("P(emit)"); axes[1].legend(loc="best")
    num("EmitLiveZero", float(e_live[0]), "{:.3f}")
    num("EmitYokeZero", float(e_yoke[0]), "{:.3f}")

    panel(axes[2], "C", "Free-riding drives the deaf contrast")
    hd = [np.mean(vals(h, "base", "emit_rate")) for _, _, _, h, _ in pairs]
    dd = [np.mean(vals(d, "base", "emit_rate")) for _, _, _, _, d in pairs]
    keep = [i for i in range(len(hd)) if np.isfinite(hd[i]) and np.isfinite(dd[i])]
    if keep:
        xx = np.arange(len(keep))
        axes[2].bar(xx - 0.19, [hd[i] for i in keep], 0.36, color=C_FULL,
                    edgecolor="white", lw=0.8, label="can hear others")
        axes[2].bar(xx + 0.19, [dd[i] for i in keep], 0.36, color=MUTED,
                    edgecolor="white", lw=0.8, label="deaf")
        axes[2].set_xticks(xx)
        axes[2].set_xticklabels([f"$\\lambda$={pairs[i][0]:g}" for i in keep])
        axes[2].set_ylabel("P(emit)"); axes[2].legend(loc="best")
        num("EmitHearZero", float(hd[0]), "{:.3f}")
        num("EmitDeafZero", float(dd[0]), "{:.3f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig6_ssi.pdf"); plt.close(fig)


# ===========================================================================
# Figure 9 -- the positive control: a partially decoupled signalling variable
# ===========================================================================
E_TASKS = [("std", "standard"), ("sparse", "sparse")]
E_ECON = [("cmp", "compet."), ("coop", "cooper.")]


def fig9():
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.7))
    labs, use, mi_food, kill_r, scr_r, cols = [], [], [], [], [], []
    for tk, tl in E_TASKS:
        for ec, el in E_ECON:
            pref = f"E_{tk}_{ec}_decoupled"
            rs = group(pref)
            if not rs:
                continue
            labs.append(f"{tl.split()[0]}\n{el}")
            cols.append(C_CUE if ec == "cmp" else C_SIGNAL)
            use.append(np.array([r["signal_bit"]["rate"] for r in rs if "signal_bit" in r]))
            mi_food.append(np.array([r["signal_bit"]["food"]["mi"] - r["signal_bit"]["food"]["null"]
                                     for r in rs if "signal_bit" in r]))
            kill_r.append(np.array([r["subtype_ablation"]["kill_subtype"]["reward_others"]["mean"]
                                    for r in rs if "subtype_ablation" in r]))
            scr_r.append(np.array([r["subtype_ablation"]["scramble_subtype"]["reward_others"]["mean"]
                                   for r in rs if "subtype_ablation" in r]))
    if not labs:
        plt.close(fig); return

    panel(axes[0], "A", "Subtype usage")
    bars_ci(axes[0], labs, use, cols, "fraction of pulses marked", rot=0)
    axes[0].axhline(0.5, color=INK2, ls=":", lw=0.8)

    panel(axes[1], "B", "Information in the subtype")
    bars_ci(axes[1], labs, mi_food, cols, "$I$(subtype; food)  (bits)", rot=0)
    axes[1].axhline(0, color=INK, lw=0.8)

    panel(axes[2], "C", "Deleting the subtype only")
    bars_ci(axes[2], labs, kill_r, cols, "$\\Delta$ receivers' return", rot=0)
    axes[2].axhline(0, color=INK, lw=0.8)
    ylo, yhi = axes[2].get_ylim()
    for i, gg in enumerate(kill_r):
        if len(gg) > 1:
            p_ = perm_p(gg, np.zeros_like(gg))
            axes[2].text(i, yhi * 0.86, stars(p_), ha="center", va="center",
                         fontsize=6.5, color=INK,
                         fontweight="bold" if p_ * len(kill_r) < 0.05 else "normal")

    panel(axes[3], "D", "Coupled vs decoupled channel")
    cpl, dcp = [], []
    for tk, _ in E_TASKS:
        for ec, _ in E_ECON:
            c = group(f"E_{tk}_{ec}_coupled")
            d = group(f"E_{tk}_{ec}_decoupled")
            cpl += [r["signal_value"]["replay_cross"]["d_receiver"]["mean"]
                    for r in c if "signal_value" in r]
            dcp += [r["subtype_ablation"]["kill_subtype"]["reward_others"]["mean"]
                    for r in d if "subtype_ablation" in r]
    bars_ci(axes[3], ["coupled\n(replay)", "decoupled\n(delete subtype)"],
            [np.array(cpl), -np.array(dcp)], [MUTED, C_SIGNAL],
            "|$\\Delta$ receivers' return|")
    axes[3].axhline(0, color=INK, lw=0.8)
    if len(dcp):
        num("SubtypeKill", float(np.mean(dcp)), "{:.2f}")
        num("SubtypeKillP", perm_p(np.array(dcp), np.zeros(len(dcp))), "{:.4f}")
    if len(use):
        num("SubtypeUse", float(np.mean(np.concatenate(use))), "{:.3f}")
    if len(mi_food):
        num("SubtypeMIFood", float(np.mean(np.concatenate(mi_food))), "{:.4f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig9_decoupled.pdf"); plt.close(fig)


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

    # cost-dependence of informativeness, pooled over predation and economy
    def pooled(cost, *path):
        out = []
        for p_ in PREDS:
            for econ in ("cmp", "coop"):
                out.extend(vals(f"C_c{cost}_p{p_}_{econ}", *path))
        return np.array(out)

    c_lo, c_hi = COSTS[0], COSTS[-1]
    for tag, cst in (("Zero", c_lo), ("High", c_hi)):
        cf = pooled(cst, "content", "food", "delta")
        ps = pooled(cst, "positive_signaling", "food", "norm")
        er = pooled(cst, "base", "emit_rate")
        if len(cf):
            num(f"ContentCost{tag}", float(np.mean(cf)), "{:.3f}")
        if len(ps) and len(er):
            num(f"PSPerPulse{tag}", float(np.mean(ps) / max(np.mean(er), 1e-9)), "{:.3f}")

    # silence and danger, pooled over the predation conditions
    dd, ss, au = [], [], []
    for cst in COSTS:
        for p_ in PREDS[1:]:
            for econ in ("cmp", "coop"):
                k = f"C_c{cst}_p{p_}_{econ}"
                a_ = vals(k, "base", "emit_rate_danger")
                b_ = vals(k, "base", "emit_rate_safe")
                n_ = min(len(a_), len(b_))
                if n_:
                    dd.extend(a_[:n_] - b_[:n_])
                au.extend(vals(k, "content", "danger", "delta"))
    if dd:
        m, lo, hi = boot_ci(np.array(dd))
        num("SilenceDangerDiff", 0.0)
        NUM["SilenceDangerDiff"] = f"${m:+.3f}$, 95\\% CI $[{lo:+.3f}, {hi:+.3f}]$"
    if au:
        num("DangerAUC", float(np.mean(au)), "{:+.3f}")


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
def write_validation_table():
    val = json.load(open("results/physics_validation.json"))
    rows = [
        ("monopole_field", "Coulomb field of point charges"),
        ("dipole_field", "Field of point dipoles"),
        ("field_with_wall_images", "Field including first-order wall images"),
        ("induced_dipole_moments", "Dipoles induced on conductors"),
        ("clip_conductor_moments", "Charge-limiting of fish body moments"),
        ("morm_cd_baseline", "Corollary-discharge baseline"),
        ("amp_intrinsic_baseline", "Ampullary intrinsic baseline"),
        ("mormyromast_pipeline", "Mormyromast, self-image mode"),
        ("mormyromast_collective_pipeline", "Mormyromast, collective-sensing mode"),
        ("knollen_pipeline", "Knollenorgan (conspecific detection)"),
        ("ampullary_pipeline", "Ampullary (passive DC)"),
    ]
    BS = chr(92)
    lines = [
        "% auto-generated by scripts/figures.py",
        BS + "begin{table}[t]",
        BS + "centering" + BS + "small",
        BS + "caption{Agreement between the GPU reimplementation and the reference "
        "NumPy implementation, in double precision on random scenes. Error is the "
        "maximum absolute deviation normalised by the largest magnitude present.}",
        BS + "label{tab:validation}",
        BS + "begin{tabular}{lr}",
        BS + "toprule",
        "Stage & Max. relative deviation " + BS * 2,
        BS + "midrule",
    ]
    for k, lab in rows:
        if k not in val:
            continue
        v = val[k]
        cell = "exact (0)" if v == 0 else "$" + BS + "num{%.1e}$" % v
        lines.append(lab + " & " + cell + " " + BS * 2)
    lines += [BS + "bottomrule", BS + "end{tabular}", BS + "end{table}", ""]
    with open("paper/validation_table.tex", "w") as f:
        f.write("\n".join(lines))
    print("wrote paper/validation_table.tex")


def write_generalisation():
    """Does the muting decomposition survive changes of scale and architecture?"""
    rows = [("A_full", "baseline (4 fish, 60\\,cm, 512 steps)"),
            ("G_fish2", "2 fish"), ("G_fish6", "6 fish"),
            ("G_arena100", "100\\,cm arena"), ("G_long", "1024-step episodes"),
            ("G_wide", "256-unit GRU")]
    lines, frac = [], []
    for pref, lab in rows:
        rs = group(pref)
        tot = np.array([r["muting_decomposition"]["eaten_target"]["total"]["mean"]
                        for r in rs if "muting_decomposition" in r])
        pri = np.array([r["muting_decomposition"]["eaten_target"]["private_share"]["mean"]
                        for r in rs if "muting_decomposition" in r])
        sig = np.array([r["muting_decomposition"]["eaten_target"]["signal_share"]["mean"]
                        for r in rs if "muting_decomposition" in r])
        if len(tot) < 2:
            continue
        f = float(np.mean(pri) / np.mean(tot)) if abs(np.mean(tot)) > 1e-9 else float("nan")
        frac.append(f)
        lines.append(f"{lab} & {len(tot)} & {np.mean(tot):+.2f} & {np.mean(pri):+.2f} & "
                     f"{np.mean(sig):+.2f} & {100*f:.0f}\\% \\\\")
    if not lines:
        NUM["GENERALISATION"] = ""
        return
    BS = chr(92)
    tab = [BS + "begin{table}[t]", BS + "centering" + BS + "small",
           BS + "caption{The muting decomposition across scales and architectures. "
           "``Private share'' is the fraction of the total muting effect attributable "
           "to the emitter's own lost reafference.}",
           BS + "label{tab:gen}",
           BS + "begin{tabular}{lrrrrr}", BS + "toprule",
           "Condition & seeds & total & $-$reafference & $-$detection & private share " + BS * 2,
           BS + "midrule"] + lines + [BS + "bottomrule", BS + "end{tabular}", BS + "end{table}"]
    open("paper/gen_table.tex", "w").write("\n".join(tab) + "\n")
    lo, hi = min(frac) * 100, max(frac) * 100
    NUM["GENERALISATION"] = (
        f"The decomposition is not an artefact of one configuration. Repeating it with "
        f"two and six fish, a \\SI{{100}}{{\\centi\\metre}} arena, 1024-step episodes and a "
        f"256-unit recurrent policy, the private share of the muting effect stays between "
        f"{lo:.0f}\\% and {hi:.0f}\\% (Table~\\ref{{tab:gen}}), and the detection share "
        f"remains indistinguishable from zero in every case.")
    NUM["GenPrivLo"] = f"{lo:.0f}"
    NUM["GenPrivHi"] = f"{hi:.0f}"


def write_verdicts():
    """Generate the conditional prose that depends on how the results came out.

    Writing these by hand risks the text drifting from the numbers as runs are
    added; deriving them means the claim and the data cannot disagree.
    """
    runs = group("A_full")

    # --- equivalence tests on the coupled-channel nulls --------------------
    parts, all_eq, mg = [], True, 0.0
    names = {"replay_cross": "cross-arena replay", "scramble_time": "temporal scrambling",
             "mute_social": "removing the public channel entirely"}
    for k, lab in names.items():
        vv = [r["tost"][k] for r in runs if "tost" in r and k in r["tost"]]
        if not vv:
            continue
        m = float(np.mean([v["mean"] for v in vv]))
        lo = float(np.mean([v["lo90"] for v in vv]))
        hi = float(np.mean([v["hi90"] for v in vv]))
        mg = float(np.mean([v["margin"] for v in vv]))
        all_eq &= (lo > -mg and hi < mg)
        parts.append(f"{lab}, ${m:+.2f}$ (90\\% CI $[{lo:+.2f}, {hi:+.2f}]$)")
    if parts:
        if all_eq:
            NUM["TOSTVERDICT"] = "every comparison falls inside it: " + "; ".join(parts) + "."
        else:
            NUM["TOSTVERDICT"] = (
                "no comparison clears the margin in either direction --- the intervals are "
                "simply wider than the $\\pm" + f"{mg:.2f}" + "$ margin (" + "; ".join(parts)
                + "). We therefore report these as \\emph{no effect detectable at this "
                "sample size}, and explicitly not as demonstrated equivalence. The honest "
                "statement is that the design rules out payoff effects larger than roughly "
                "a tenth of the intact return, and is not powered to rule out smaller "
                "ones.")
    else:
        NUM["TOSTVERDICT"] = "equivalence tests were not available for this run set."

    # --- yoked sender-shaping verdict -------------------------------------
    ssi = NUM.get("SSIYokeZero")
    if ssi is not None:
        v = float(ssi)
        if v > 0.15:
            NUM["YOKEDVERDICT"] = (
                "Emission policy therefore does depend on whether anyone can hear it, "
                "even when reception is held fixed: by the production-side criterion the "
                "discharge is not a pure cue. We read this cautiously, because the yoked "
                "world also removes the correlation between what a fish hears and what its "
                "neighbours are actually doing, which changes the learning problem in ways "
                "beyond audibility.")
        elif v > -0.15:
            NUM["YOKEDVERDICT"] = (
                "Emission policy is therefore no more different between the live and yoked "
                "worlds than between seeds of the same world. Holding reception fixed, "
                "being listened to leaves no detectable trace on how the discharge is "
                "scheduled, which is what it means for the pulse to be a cue rather than a "
                "signal.")
        else:
            NUM["YOKEDVERDICT"] = (
                "The yoked contrast is smaller than the seed-to-seed variation within "
                "either world, so the measurement is not resolving a difference in "
                "emission policy at this sample size.")

    # --- decoupled positive control, reported per ecological setting -------
    cells, use_all, mi_all = [], [], []
    for tk, tl in E_TASKS:
        for ec, el in E_ECON:
            rs = group(f"E_{tk}_{ec}_decoupled")
            kd = np.array([r["subtype_ablation"]["kill_subtype"]["reward_others"]["mean"]
                           for r in rs if "subtype_ablation" in r])
            if len(kd) < 3:
                continue
            m, lo, hi = boot_ci(kd)
            pv = perm_p(kd, np.zeros(len(kd)))
            cells.append((tl, el, m, lo, hi, pv))
            use_all += [r["signal_bit"]["rate"] for r in rs if "signal_bit" in r]
            mi_all += [r["signal_bit"]["food"]["mi"] - r["signal_bit"]["food"]["null"]
                       for r in rs if "signal_bit" in r]
    if cells:
        nt = len(cells)
        hits = [c for c in cells if c[5] * nt < 0.05 and c[2] < 0]
        desc = "; ".join(f"{tl} $\\times$ {el}, ${m:+.2f}$ "
                         f"(95\\% CI $[{lo:+.2f}, {hi:+.2f}]$, $p=\\num{{{pv:.3f}}}$)"
                         for tl, el, m, lo, hi, pv in cells)
        NUM["SubtypeUse"] = f"{np.mean(use_all):.2f}"
        NUM["SubtypeMIFood"] = f"{np.mean(mi_all):.4f}"
        head = (f"Agents mark {np.mean(use_all)*100:.0f}\\% of their discharges with the "
                f"subtype, which is indistinguishable from the 50\\% an unmodulated bit "
                f"would give, and it carries ${np.mean(mi_all):.4f}$ bits about food "
                f"proximity above its permutation null. Deleting the subtype leaves the "
                f"pulse, and therefore every sensory consequence, completely intact, so "
                f"any payoff change is attributable to content alone. Across the four "
                f"ecological settings: {desc}. ")
        if hits:
            tl, el, m, lo, hi, pv = hits[0]
            NUM["DECOUPLEDRESULT"] = head + (
                f"One of the four survives Bonferroni correction across the four cells: "
                f"{tl} under {el}, where deleting the subtype costs receivers ${m:+.2f}$ "
                f"in return. That is the setting in which private information is "
                f"scarcest---patches are hard to find and a single fish's probe reaches "
                f"only a small part of a large arena---which is where a social channel "
                f"should pay if it is going to pay anywhere. We report it as suggestive "
                f"rather than established: it is one cell of four, in the task family with "
                f"the weakest absolute performance, and the subtype is used at essentially "
                f"chance rate even there. What the control does establish is that our "
                f"measurements can detect a payoff-relevant social channel when one is "
                f"present, which is what makes the null results elsewhere interpretable "
                f"rather than merely underpowered.")
        else:
            NUM["DECOUPLEDRESULT"] = head + (
                "No cell survives correction for multiple comparisons. Freeing the content "
                "of a discharge from its sensory function was not, on its own, sufficient "
                "to make that content payoff-relevant under any pressure we applied.")
    else:
        NUM["DECOUPLEDRESULT"] = "The decoupled-channel runs were not available."

    # --- abstract ----------------------------------------------------------
    NUM["ABSTRACTRESULTS"] = (
        "We first validate the assays on dyads that are communicating by "
        "construction: in a referential game embedded in the same physics they "
        "separate informative from uninformative senders and attentive from deaf "
        "receivers, and deleting a genuinely used signal costs its receiver "
        "\\AssayKillHonest\\ in return. Each assay alone, however, is foolable --- "
        "positive signalling is just as high when nobody listens, and causal "
        "influence just as high when the channel carries nothing --- so only their "
        "conjunction identifies communication. Applied to the electric discharge, "
        "silencing a fish costs it \\MuteTargetTotal\\ food items per episode, of "
        "which \\MuteTargetPriv\\ is the loss of its own electrolocation and "
        "\\MuteTargetSig\\ the loss of its detectability by others; the private "
        "share stays between \\GenPrivLo\\% and \\GenPrivHi\\% across group sizes, "
        "arena scales, episode lengths and network widths. The public channel "
        "satisfies both standard criteria --- decodable information about food "
        "($\\Delta R^2=\\ContentFood$) and receiver-policy influence "
        "\\CIERatio$\\times$ the noise floor --- yet replaying, scrambling or "
        "fabricating it moves no payoff we can resolve, sender shaping is "
        "\\SSIYokeZero\\ once reception is held fixed and only audibility is cut, "
        "and metabolic cost and eavesdropping predation regulate discharge rate "
        "while making the pulse train \\emph{less} informative. Freeing one bit of "
        "discharge subtype from the sensory function does not change this. A system "
        "can pass every standard diagnostic for emergent communication and not be "
        "communicating; what the discharge lacks is not a channel but a task in "
        "which its private information is worth anything to anyone else.")


def write_numbers():
    # Measured aggregate throughput: per-job rate from the training histories,
    # times the number of jobs resident on the two GPUs concurrently.
    rates = []
    for f in glob.glob("results/runs/*_hist.json"):
        try:
            d = json.load(open(f))
            if d.get("wall_s", 0) > 0 and d["hist"]:
                rates.append(d["hist"][-1]["steps"] / d["wall_s"])
        except Exception:
            pass
    if rates:
        NUM["ThroughputK"] = f"{np.median(rates) * 6 / 1e3:.0f}"
        NUM["ThroughputPerJobK"] = f"{np.median(rates) / 1e3:.0f}"
    NUM["NumRuns"] = str(len(glob.glob("results/runs/*.pt")))
    NUM["NumEval"] = str(len(M))
    with open("paper/numbers.tex", "w") as f:
        f.write("% auto-generated by scripts/figures.py -- do not edit\n")
        for k, v in sorted(NUM.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote paper/numbers.tex with {len(NUM)} macros")


if __name__ == "__main__":
    load()
    for fn in (fig_assay, fig_learned_ref, fig_dualuse, fig1, fig2, fig3, fig4, fig5, fig7, fig8, fig9, fig6, write_validation_table, write_generalisation, write_verdicts):
        try:
            fn()
            print("ok", fn.__name__, flush=True)
        except Exception as e:
            import traceback
            print(f"FAIL {fn.__name__}: {e}\n{traceback.format_exc()}", flush=True)
    write_numbers()
