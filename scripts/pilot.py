"""Pilot gate E0: does the discharge have private sensory value, and how long
does the foraging policy take to converge?

Trains three conditions from scratch:
  full      -- normal agents (probe + public signal)
  silent    -- discharge permanently disabled (passive ampullary sensing only)
  deaf      -- discharge intact, knollen channel disabled (no social reception)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from eodcomm.train import train_one  # noqa: E402

CONDS = {
    "full": {},
    "silent": {"eod_allowed": False},
    "deaf": {"knollen_enabled": False},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(CONDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60_000_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="results/pilot")
    a = ap.parse_args()

    base = dict(
        n_fish=4, n_food=48, n_patches=4, episode_len=512, batch=a.batch,
        arena_cm=(60.0, 60.0),
    )
    base.update(CONDS[a.cond])
    tag = f"{a.cond}_s{a.seed}"
    _, _, hist = train_one(
        base, seed=a.seed, total_steps=a.steps, out_dir=a.out, tag=tag,
        ppo_kwargs={"rollout": 64},
    )
    print(json.dumps(hist[-1]))
