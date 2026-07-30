"""Define and launch the full experimental programme."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from run_jobs import run_pool  # noqa: E402

BASE = dict(n_fish=4, n_food=48, n_patches=4, episode_len=512, batch=512, arena_cm=[60.0, 60.0])
STEPS = 18_000_000
OUT = "results/runs"
SEEDS_A = [0, 1, 2, 3, 4, 5]
SEEDS_C = [0, 1, 2]

jobs = []


def add(name, seeds, **over):
    for s in seeds:
        ek = dict(BASE)
        ek.update(over)
        jobs.append({"name": f"{name}_s{s}", "env_kwargs": ek, "seed": s, "steps": STEPS, "out": OUT})


# --- Set A: three-way channel decomposition, trained from scratch -----------
# A single discharge has three separable consequences: it illuminates the
# emitter's own world (reafference), it illuminates its neighbours' world
# (an exploitable cue), and it is detected by their knollenorgans (a candidate
# signal).  Upstream's muting removes all three at once; these conditions
# remove them one at a time.
add("A_full", SEEDS_A)
add("A_noknollen", SEEDS_A, knollen_enabled=False)                       # no detection
add("A_noillum", SEEDS_A, illuminate_others=False)                       # no shared illumination
add("A_private", SEEDS_A, knollen_enabled=False, illuminate_others=False)  # pure private probe
add("A_noself", SEEDS_A, collective_sensing=2)                           # no reafference
add("A_silent", SEEDS_A, eod_allowed=False)                              # no discharge at all

# --- Set B: Sender Shaping Index vs metabolic cost --------------------------
# hearing vs deaf worlds at matched cost; cost 0 reuses A_full / A_noknollen.
for c in (0.02, 0.06):
    add(f"B_hear_c{c}", SEEDS_A, eod_cost=c)
    add(f"B_deaf_c{c}", SEEDS_A, eod_cost=c, knollen_enabled=False)

# --- Set C: the evolutionary phase grid -------------------------------------
for c in (0.0, 0.01, 0.02, 0.04, 0.08):
    for p in (0.0, 5.0, 20.0):
        for econ, share in (("cmp", 0.0), ("coop", 1.0)):
            add(
                f"C_c{c}_p{p}_{econ}", SEEDS_C,
                eod_cost=c, predation=p, shared_food=share,
                pred_detect_cm=60.0, pred_speed_cm=0.45,
            )

# --- Set D: identity persistence and social range ---------------------------
for c in (0.0, 0.04):
    add(f"D_idshuf_c{c}", SEEDS_C, eod_cost=c, persistent_identity=False, size_mode="uniform")
for g in (0.03, 0.003):
    for c in (0.0, 0.04):
        add(f"D_range{g}_c{c}", SEEDS_C, eod_cost=c, knollen_gain=g)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    sel = [j for j in jobs if (only is None or j["name"].startswith(only))]
    print(f"{len(sel)} jobs, {STEPS/1e6:.0f}M steps each", flush=True)
    os.makedirs(OUT, exist_ok=True)
    done = run_pool(sel, gpus=(0, 1), per_gpu=3, logdir="logs/runs")
    with open("results/run_manifest.json", "w") as f:
        json.dump({"jobs": sel, "done": done}, f, indent=2)
    nfail = sum(1 for d in done if not d["ok"])
    print(f"\ncomplete: {len(done)-nfail} ok, {nfail} failed")
