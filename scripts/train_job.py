"""Single training job (invoked as a subprocess by run_jobs.py)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from eodcomm.train import train_one  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=25_000_000)
    ap.add_argument("--out", default="results/runs")
    a = ap.parse_args()

    ek = json.loads(a.env)
    if "arena_cm" in ek:
        ek["arena_cm"] = tuple(ek["arena_cm"])
    train_one(ek, seed=a.seed, total_steps=a.steps, out_dir=a.out, tag=a.name,
              ppo_kwargs={"rollout": 64}, log_every=16)
