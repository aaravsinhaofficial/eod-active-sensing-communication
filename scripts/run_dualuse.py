"""Replicate the channel decomposition in a world with no electric fish in it.

Trains the same five channel conditions in the minimal dual-use environment,
then runs the same frozen-policy intervention battery, so the decomposition can
be compared directly with the electric-fish result.
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from eodcomm.dualuse import DualUseConfig, DualUseEnv  # noqa: E402
from eodcomm.ppo import MAPPO, PPOConfig  # noqa: E402
from eodcomm.env import ChannelSpec  # noqa: E402
from eodcomm.metrics import collect, causal_influence  # noqa: E402
from eodcomm.interventions import paired_diff  # noqa: E402

CONDS = {
    "full": {},
    "no_detect": {"detection": False},
    "no_illum": {"illuminate_others": False},
    "private": {"detection": False, "illuminate_others": False},
    "silent": {"ping_allowed": False},
}
SEEDS = [0, 1, 2, 3, 4, 5]
STEPS = 20_000_000
COSTS = [0.0, 0.02, 0.05, 0.1]


def train(over, seed, steps=STEPS, batch=512):
    cfg = DualUseConfig(batch=batch, **over)
    env = DualUseEnv(cfg)
    obs = env.reset(seed)
    tr = MAPPO(env, PPOConfig(rollout=64), seed=seed)
    per = 64 * batch * env.F
    for _ in range(max(1, steps // per)):
        obs, buf, lv, _ = tr.rollout(obs)
        tr.update(buf, lv)
    return tr, env


@torch.no_grad()
def battery(tr, env, steps=384, seed=7, target=0):
    F = env.F
    others = [j for j in range(F) if j != target]
    res = {}
    specs = {
        "intact": None,
        "mute_both": ChannelSpec(mode="mute", agents=(target,)),
        "mute_self": ChannelSpec(mode="social", agents=(target,)),
        "mute_illum": ChannelSpec(mode="signal_only", agents=(target,)),
        "mute_detect": ChannelSpec(mode="cue_only", agents=(target,)),
    }
    for k, sp in specs.items():
        rec = collect(tr, env, steps, channel=sp, seed=seed)
        res[k] = {"_per_arena": {
            "collected_target": rec["ate"][:, :, target].sum(0).cpu().numpy(),
            "collected_others": rec["ate"][:, :, others].sum(0).mean(-1).cpu().numpy(),
            "reward_others": rec["rew"][:, :, others].sum(0).mean(-1).cpu().numpy(),
        }}
    out = {}
    for dv in ("collected_target", "collected_others"):
        out[dv] = {k: paired_diff(res, k, "intact", dv)
                   for k in ("mute_both", "mute_self", "mute_illum", "mute_detect")}
    return out


if __name__ == "__main__":
    rows = {}
    # --- channel decomposition ---------------------------------------------
    for name, over in CONDS.items():
        got, pings = [], []
        for s in SEEDS:
            tr, env = train(over, s)
            rec = collect(tr, env, 384, seed=100 + s)
            got.append(float(rec["ate"].sum(0).mean()))
            pings.append(float(rec["emit"].mean()))
            if name == "full":
                b = battery(tr, env, seed=7 + s)
                rows.setdefault("decomp", []).append(b)
                ci, cn = causal_influence(tr, env, n_steps=96, seed=3 + s)
                rows.setdefault("cie", []).append(
                    {"mean": float(np.nanmean(ci)), "null": float(np.nanmean(cn))})
            del tr, env
            torch.cuda.empty_cache()
        rows.setdefault("cond", {})[name] = {"collected": got, "ping": pings}
        print(f"[{name:10s}] collected={np.mean(got):.2f}+-{np.std(got):.2f} "
              f"ping={np.mean(pings):.3f}", flush=True)

    # --- does cost make the channel less informative here too? --------------
    for c in COSTS:
        er, coll = [], []
        for s in SEEDS[:3]:
            tr, env = train({"ping_cost": c}, s)
            rec = collect(tr, env, 384, seed=200 + s)
            er.append(float(rec["emit"].mean()))
            coll.append(float(rec["ate"].sum(0).mean()))
            del tr, env
            torch.cuda.empty_cache()
        rows.setdefault("cost", {})[str(c)] = {"ping": er, "collected": coll}
        print(f"[cost {c:<5}] ping={np.mean(er):.3f} collected={np.mean(coll):.2f}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/dualuse.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote results/dualuse.json")
