"""Evaluate every trained run and write a tidy metric table.

For each checkpoint this computes
  * training-condition summaries (food, emission rate, mortality, spacing),
  * the counterfactual channel battery on the frozen policy,
  * interventional causal influence, positive listening, positive signalling,
  * what a receiver can decode from the pulse train about food/danger/dominance/movement,
  * the honesty / manipulation classification from the payoff consequences of
    the signal's contingency.
"""

import argparse
import glob
import json
import os
import sys
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from eodcomm.train import load_agent  # noqa: E402
from eodcomm.metrics import (  # noqa: E402
    causal_influence, collect, decode_content, positive_listening, positive_signaling,
    signal_bit_metrics,
)
from eodcomm.interventions import (  # noqa: E402
    channel_battery, decompose_muting, paired_diff, signal_value, tost_equivalence,
)


def eval_run(path, batch=192, steps=512, seed=7, do_channels=True):
    tr, env, ck = load_agent(path, env_overrides=dict(batch=batch))
    name = os.path.basename(path)[:-3]
    out = {"name": name, "seed": ck["seed"], "env": {k: (list(v) if isinstance(v, tuple) else v)
                                                     for k, v in ck["env_kwargs"].items()}}

    rec = collect(tr, env, steps, seed=seed)
    F = env.F
    pos = rec["pos"]
    d = torch.cdist(pos, pos)
    eye = torch.eye(F, device=d.device, dtype=torch.bool)
    nn = d.masked_fill(eye[None, None], float("inf")).min(-1).values

    # emission rate by dominance rank (submissive silence)
    size = rec["size"][0, 0]
    rank = torch.argsort(torch.argsort(size))
    er = rec["emit"].mean(dim=(0, 1))
    lo = er[rank < F // 2].mean().item()
    hi = er[rank >= F // 2].mean().item()

    out["base"] = {
        "eaten_per_ep": rec["ate"].sum(0).mean().item(),
        "eaten_group": rec["ate"].sum(0).sum(-1).mean().item(),
        "reward_per_ep": rec["rew"].sum(0).mean().item(),
        "emit_rate": rec["emit"].mean().item(),
        "emit_rate_subordinate": lo,
        "emit_rate_dominant": hi,
        "silence_index": 1.0 - lo / max(hi, 1e-9),
        "nn_dist": nn.mean().item(),
        "struck_per_ep": rec["struck"].sum(0).mean().item(),
        "bit_per_ep": rec["bit"].sum(0).mean().item(),
        "heard_frac": rec["heard"].mean().item(),
        "emit_rate_food": rec["emit"][rec["food_near10"] > 0].mean().item(),
        "emit_rate_nofood": rec["emit"][rec["food_near10"] == 0].mean().item(),
    }
    pd_ = rec["pred_dist"]
    if torch.isfinite(pd_).any() and (pd_ < 1e5).any():
        near = pd_ < 25
        out["base"]["emit_rate_danger"] = rec["emit"][near].mean().item() if near.any() else float("nan")
        out["base"]["emit_rate_safe"] = rec["emit"][~near].mean().item()

    out["positive_signaling"] = positive_signaling(rec)
    sb = signal_bit_metrics(rec)
    if sb is not None:
        out["signal_bit"] = sb
    out["content"] = decode_content(rec)

    if do_channels:
        ci, ci_null = causal_influence(tr, env, n_steps=128, seed=seed + 1)
        out["cie"] = {"mean": float(np.nanmean(ci)), "null": float(np.nanmean(ci_null)),
                      "matrix": np.nan_to_num(ci).tolist()}
        pl, pl_zero = positive_listening(tr, env, n_steps=128, seed=seed + 2)
        out["positive_listening"] = {"shift_null": pl, "zero_null": pl_zero}
        res, _ = channel_battery(tr, env, n_steps=steps, seed=seed)
        out["muting_decomposition"] = decompose_muting(res)
        out["signal_value"] = signal_value(res)
        # The cross-arena replay transmits the train the same policy produced in
        # an independently drawn world while this world is held fixed: that is
        # exactly the natural indirect effect, and unlike deletion it keeps the
        # message inside its own marginal.
        out["nie"] = {
            "receivers": paired_diff(res, "replay_cross", "intact", "reward_others"),
            "sender": paired_diff(res, "replay_cross", "intact", "reward_target"),
            "context_swapped": paired_diff(res, "replay_context", "intact", "reward_others"),
        }
        # phantom dose-response and the rest of the battery, minus the bulky
        # per-arena arrays that were only needed for the paired statistics
        out["channels"] = {k: {kk: vv for kk, vv in v.items() if kk != "_per_arena"}
                           for k, v in res.items()}
        if "kill_subtype" in res:
            # the intact pulse, only the decoupled content removed
            base_r = abs(res["intact"]["reward_others"]) * 0.05 + 1e-9
            out["subtype_ablation"] = {
                k: {
                    "reward_others": paired_diff(res, k, "intact", "reward_others"),
                    "reward_target": paired_diff(res, k, "intact", "reward_target"),
                    "eaten_group": paired_diff(res, k, "intact", "eaten_group"),
                    "tost_reward_others": tost_equivalence(res, k, "intact",
                                                           "reward_others", base_r),
                }
                for k in ("kill_subtype", "scramble_subtype")
            }
        # equivalence tests on the coupled-channel nulls, against a 5% margin
        marg = abs(res["intact"]["reward_others"]) * 0.05 + 1e-9
        out["tost"] = {
            k: tost_equivalence(res, k, "intact", "reward_others", marg)
            for k in ("replay_cross", "scramble_time", "mute_social") if k in res
        }
        out["phantom_dose"] = {
            k: paired_diff(res, k, "intact", "eaten_others")
            for k in res if k.startswith("phantom_")
        }
    del tr, env
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/runs/*.pt")
    ap.add_argument("--out", default="results/metrics")
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--light", action="store_true", help="skip the channel battery (phase-grid runs)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(a.glob))
    paths = [p for i, p in enumerate(paths) if i % a.nshard == a.shard]
    print(f"shard {a.shard}/{a.nshard}: {len(paths)} runs", flush=True)
    for i, p in enumerate(paths):
        name = os.path.basename(p)[:-3]
        dest = os.path.join(a.out, name + ".json")
        if os.path.exists(dest):
            continue
        # the channel battery is only meaningful where a social channel exists
        heavy = not a.light and not name.startswith("C_")
        try:
            r = eval_run(p, batch=a.batch, steps=a.steps, do_channels=heavy)
            with open(dest, "w") as f:
                json.dump(r, f, indent=1)
            print(f"[{i+1}/{len(paths)}] {name} eaten={r['base']['eaten_per_ep']:.2f} "
                  f"emit={r['base']['emit_rate']:.3f}", flush=True)
        except Exception:
            print(f"[FAIL] {name}\n{traceback.format_exc()}", flush=True)
