"""Generalisation grid, and more power for the one decoupled cell that moved.

G: does the muting decomposition survive changes of group size, arena scale,
   episode length and network width?
H: additional seeds for E_sparse_cmp_decoupled and its coupled control, so the
   one cell with an effect either survives correction or is reported as null.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from run_jobs import run_pool  # noqa: E402

OUT = "results/runs"
STEPS = 18_000_000
STD = dict(n_fish=4, n_food=48, n_patches=4, episode_len=512, batch=512,
           arena_cm=[60.0, 60.0])
SPARSE = dict(n_fish=4, n_food=40, n_patches=2, patch_sigma_cm=5.0,
              episode_len=512, batch=512, arena_cm=[100.0, 100.0])

jobs = []


def add(name, base, seeds, steps=STEPS, ppo=None, **over):
    for s in seeds:
        ek = dict(base)
        ek.update(over)
        j = {"name": f"{name}_s{s}", "env_kwargs": ek, "seed": s, "steps": steps, "out": OUT}
        if ppo:
            j["ppo"] = ppo
        jobs.append(j)


S6 = [0, 1, 2, 3, 4, 5]

# --- G: generalisation of the headline decomposition ------------------------
add("G_fish2", STD, S6, n_fish=2, n_food=24)
add("G_fish6", STD, S6, n_fish=6, n_food=72)
add("G_arena100", STD, S6, arena_cm=[100.0, 100.0], n_food=48)
add("G_long", STD, S6, episode_len=1024, steps=26_000_000)
add("G_wide", STD, S6, ppo={"gru": 256, "hidden": 256})

# --- H: power for the decoupled cell that showed an effect ------------------
S6b = [6, 7, 8, 9, 10, 11]
add("E_sparse_cmp_decoupled", SPARSE, S6b, steps=25_000_000, shared_food=0.0, signal_channel=True)
add("E_sparse_cmp_coupled", SPARSE, S6b, steps=25_000_000, shared_food=0.0)
add("E_sparse_coop_decoupled", SPARSE, S6b, steps=25_000_000, shared_food=1.0, signal_channel=True)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    sel = [j for j in jobs if (only is None or j["name"].startswith(only))]
    print(f"{len(sel)} jobs", flush=True)
    done = run_pool(sel, gpus=(0, 1), per_gpu=3, logdir="logs/runs")
    nfail = sum(1 for d in done if not d["ok"])
    print(f"complete: {len(done)-nfail} ok, {nfail} failed")
