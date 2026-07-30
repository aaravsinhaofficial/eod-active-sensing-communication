"""Can policy gradient invent the code, or only read one?

R_emergent   both sides learn: the sender must discover that conditioning its
             subtype on the private cue pays, and the receiver must discover
             that the subtype is worth reading.  If any seed solves this, the
             nulls in the foraging world cannot be blamed on the optimiser.
R_recvonly   the sender is scripted honest; only the receiver learns.  Isolates
             which half of the chicken-and-egg problem is the hard one.
R_sendonly   the receiver is scripted attentive; only the sender learns.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from run_jobs import run_pool  # noqa: E402

OUT = "results/runs"
STEPS = 50_000_000
SEEDS = list(range(12))

REF = dict(n_fish=2, n_food=10, task="referential", signal_channel=True,
           arena_cm=[90.0, 70.0], shared_food=1.0, episode_len=512,
           size_mode="none", predation=0.0, batch=512)

jobs = []


def add(name, seeds, **over):
    for s in seeds:
        ek = dict(REF)
        ek.update(over)
        jobs.append({"name": f"{name}_s{s}", "env_kwargs": ek, "seed": s,
                     "steps": STEPS, "out": OUT,
                     # a discharge-subtype exploration bonus; without one the
                     # signalling head collapses long before the receiver has
                     # any reason to attend to it
                     "ppo": {"ent_coef": 0.02, "ent_coef_emit": 0.05}})


add("R_emergent", SEEDS)
add("R_recvonly", SEEDS, ref_scripted="honest")

# with the Eccles positive-signalling bias on the message head
for _s in SEEDS:
    jobs.append({"name": f"R_eccles_s{_s}", "env_kwargs": dict(REF), "seed": _s,
                 "steps": STEPS, "out": OUT,
                 "ppo": {"ent_coef": 0.02, "ent_coef_emit": 0.05, "ps_bias": 0.5}})

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    sel = [j for j in jobs if (only is None or j["name"].startswith(only))]
    print(f"{len(sel)} jobs, {STEPS/1e6:.0f}M steps each", flush=True)
    done = run_pool(sel, gpus=(0, 1), per_gpu=3, logdir="logs/runs")
    nfail = sum(1 for d in done if not d["ok"])
    print(f"complete: {len(done)-nfail} ok, {nfail} failed")


