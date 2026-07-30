"""Additional seeds for the channel-decomposition conditions.

Training is bimodal -- a seed either finds patch foraging or stays at the floor
set by passive sensing alone -- so the from-scratch comparison needs more seeds
than the frozen-policy analyses do.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from run_jobs import run_pool  # noqa: E402
from launch_all import BASE, OUT, STEPS  # noqa: E402

SEEDS = [6, 7, 8, 9, 10, 11]
SCRAM_SEEDS = [0, 1, 2, 3, 4, 5]
CONDS = {
    "A_full": {},
    "A_noknollen": {"knollen_enabled": False},
    "A_noillum": {"illuminate_others": False},
    "A_private": {"knollen_enabled": False, "illuminate_others": False},
    "A_noself": {"collective_sensing": 2},
    "A_silent": {"eod_allowed": False},
    "B_hear_c0.02": {"eod_cost": 0.02},
    "B_deaf_c0.02": {"eod_cost": 0.02, "knollen_enabled": False},
    "B_hear_c0.06": {"eod_cost": 0.06},
    "B_deaf_c0.06": {"eod_cost": 0.06, "knollen_enabled": False},
    # SSI control: receivers hear, but sender identity is scrambled at train time
    "B_scram_c0.0": {"scramble_id_always": True},
    "B_scram_c0.06": {"eod_cost": 0.06, "scramble_id_always": True},
}

jobs = []
for name, over in CONDS.items():
    for s in (SCRAM_SEEDS if "scram" in name else SEEDS):
        ek = dict(BASE)
        ek.update(over)
        jobs.append({"name": f"{name}_s{s}", "env_kwargs": ek, "seed": s,
                     "steps": STEPS, "out": OUT})

if __name__ == "__main__":
    print(f"{len(jobs)} extra-seed jobs", flush=True)
    done = run_pool(jobs, gpus=(0, 1), per_gpu=3, logdir="logs/runs")
    nfail = sum(1 for d in done if not d["ok"])
    print(f"complete: {len(done)-nfail} ok, {nfail} failed")
