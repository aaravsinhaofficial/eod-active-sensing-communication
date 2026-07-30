"""Parallel job runner: trains many conditions across the available GPUs."""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)


def run_pool(jobs, gpus=(0, 1), per_gpu=2, logdir="logs", python=sys.executable):
    """jobs: list of dicts with keys name, env_kwargs, seed, steps, out."""
    os.makedirs(logdir, exist_ok=True)
    pending = list(jobs)
    running = []  # (proc, job, gpu, logfile, t0)
    slots = {g: 0 for g in gpus}
    done = []
    t_start = time.time()

    while pending or running:
        for g in gpus:
            while slots[g] < per_gpu and pending:
                j = pending.pop(0)
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(g)
                lf = os.path.join(logdir, f"{j['name']}.log")
                cmd = [
                    python, os.path.join(HERE, "train_job.py"),
                    "--name", j["name"],
                    "--env", json.dumps(j["env_kwargs"]),
                    "--seed", str(j["seed"]),
                    "--steps", str(j["steps"]),
                    "--out", j["out"],
                ]
                with open(lf, "w") as fh:
                    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                running.append((p, j, g, lf, time.time()))
                slots[g] += 1
        time.sleep(3)
        still = []
        for p, j, g, lf, t0 in running:
            if p.poll() is None:
                still.append((p, j, g, lf, t0))
            else:
                slots[g] -= 1
                ok = p.returncode == 0
                done.append({"name": j["name"], "ok": ok, "sec": time.time() - t0})
                status = "ok " if ok else "FAIL"
                print(f"[{status}] {j['name']:44s} {time.time()-t0:6.0f}s "
                      f"({len(done)}/{len(jobs)}, {time.time()-t_start:.0f}s elapsed)", flush=True)
                if not ok:
                    with open(lf) as fh:
                        print("      " + "\n      ".join(fh.read().strip().splitlines()[-6:]), flush=True)
        running = still
    return done
