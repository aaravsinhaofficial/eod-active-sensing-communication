"""The positive control: does partial decoupling let communication emerge?

Set E asks whether giving the agents a signalling variable that is decoupled
from sensing changes the answer.  A discharge subtype rides on the pulse -- so
sending still costs a pulse -- but the subtype itself alters nothing about the
emitter's own electric image, nothing about how the pulse illuminates
neighbours, and nothing about predator detectability.  Crossed with cooperative
versus competitive harvests, and with a sparse-patch task in which finding food
at all is the bottleneck and so information is worth having.

Set F is the sender-shaping control the hearing/deaf contrast could not provide.
`knollen_enabled=False` removes *reception*; the yoked condition instead keeps
reception statistics matched while severing the link between any sender's pulses
and any receiver's input, which is what isolates being listened to.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from run_jobs import run_pool  # noqa: E402

OUT = "results/runs"
SEEDS = [0, 1, 2, 3, 4, 5]
STEPS = 25_000_000

STANDARD = dict(n_fish=4, n_food=48, n_patches=4, episode_len=512, batch=512,
                arena_cm=[60.0, 60.0])
# Sparse task: patches are hard to find, so knowing where one is has real value.
SPARSE = dict(n_fish=4, n_food=40, n_patches=2, patch_sigma_cm=5.0,
              episode_len=512, batch=512, arena_cm=[100.0, 100.0])

jobs = []


def add(name, base, **over):
    for s in SEEDS:
        ek = dict(base)
        ek.update(over)
        jobs.append({"name": f"{name}_s{s}", "env_kwargs": ek, "seed": s,
                     "steps": STEPS, "out": OUT})


# --- Set E: coupled vs decoupled x competition vs cooperation x task --------
for task, base in (("std", STANDARD), ("sparse", SPARSE)):
    for econ, share in (("cmp", 0.0), ("coop", 1.0)):
        add(f"E_{task}_{econ}_coupled", base, shared_food=share)
        add(f"E_{task}_{econ}_decoupled", base, shared_food=share, signal_channel=True)

# --- Set F: yoked reception, the sender-shaping control ---------------------
for c in (0.0, 0.06):
    add(f"F_live_c{c}", STANDARD, eod_cost=c)
    add(f"F_yoked_c{c}", STANDARD, eod_cost=c, yoked_knollen=True)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    sel = [j for j in jobs if (only is None or j["name"].startswith(only))]
    print(f"{len(sel)} jobs, {STEPS/1e6:.0f}M steps each", flush=True)
    done = run_pool(sel, gpus=(0, 1), per_gpu=3, logdir="logs/runs")
    nfail = sum(1 for d in done if not d["ok"])
    print(f"complete: {len(done)-nfail} ok, {nfail} failed")
