"""Run the whole metric battery on dyads whose communication status is known.

Four conditions, all with the same physics, the same forced pulse schedule and
the same reactive receiver:

  honest+listen   sender's subtype is the private cue, receiver uses it
                  -> communication exists; every assay should fire
  random+listen   subtype is a rate-matched fair coin, receiver uses it
                  -> a channel that is attended to but carries nothing
  honest+deaf     subtype is the cue, receiver ignores it
                  -> a signal nobody listens to
  random+deaf     neither
                  -> nothing

If the assays pass the first and reject the others, then a null in the
naturalistic foraging environment means something.
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from eodcomm.env import EnvConfig, FishEnv  # noqa: E402
from eodcomm.ppo import PPOConfig  # noqa: E402
from eodcomm.scripted import ScriptedReferentialNet, ScriptedRunner  # noqa: E402
from eodcomm.metrics import (  # noqa: E402
    causal_influence, collect, positive_listening, quantize, cmi_plugin,
)
from eodcomm.interventions import evaluate_channel, paired_diff  # noqa: E402

BASE = dict(n_fish=2, n_food=10, task="referential", signal_channel=True,
            arena_cm=(90.0, 70.0), shared_food=1.0, episode_len=512,
            size_mode="none", predation=0.0)


def run_condition(script, listen, batch=256, steps=512, seed=0):
    cfg = EnvConfig(batch=batch, ref_scripted=script, **BASE)
    env = FishEnv(cfg)
    net = ScriptedReferentialNet(env.obs_dim, env.F, PPOConfig(gru=8),
                                 n_disc=env.n_disc, listen=listen)
    tr = ScriptedRunner(env, net)

    rec = collect(tr, env, steps, seed=seed)
    out = {"script": script, "listen": listen}
    out["arrivals_per_ep"] = float(rec["arrived"][:, :, 1].float().sum(0).mean())
    out["receiver_return"] = float(rec["rew"][:, :, 1].sum(0).mean())

    # --- content: what does the heard subtype say about the world? ---------
    sub = rec["obs"][:, :, 1, -6].cpu().numpy().reshape(-1)
    act = rec["active"].cpu().numpy().reshape(-1)
    z = np.zeros_like(quantize(sub, 3))
    mi = cmi_plugin(quantize(sub, 3), act.astype(np.int64), z, 3, 2, 1)
    rng = np.random.default_rng(0)
    null = [cmi_plugin(quantize(sub, 3), act[rng.permutation(act.size)].astype(np.int64), z, 3, 2, 1)
            for _ in range(50)]
    out["content_mi"] = float(mi)
    out["content_null"] = float(np.mean(null))

    # --- causal influence and positive listening ---------------------------
    ci, cn = causal_influence(tr, env, n_steps=96, seed=seed + 1)
    out["cie"] = float(np.nanmean(ci))
    out["cie_null"] = float(np.nanmean(cn))
    pl, pz = positive_listening(tr, env, n_steps=96, seed=seed + 2)
    out["pl_shift"] = float(pl)

    # --- mediation: the natural indirect effect ----------------------------
    # Resample the referent, recompute the message from it, transmit that, and
    # leave the true referent (and so every non-message pathway) untouched.  The
    # message stays inside its own marginal distribution throughout, which is
    # what distinguishes this from deleting it.
    cfg_m = EnvConfig(batch=batch, ref_scripted=script, ref_mediate=True, **BASE)
    env_m = FishEnv(cfg_m)
    net_m = ScriptedReferentialNet(env_m.obs_dim, env_m.F, PPOConfig(gru=8),
                                   n_disc=env_m.n_disc, listen=listen)
    tr_m = ScriptedRunner(env_m, net_m)
    rec_m = collect(tr_m, env_m, steps, seed=seed)
    nie = float(rec_m["rew"][:, :, 1].sum(0).mean()) - out["receiver_return"]
    a_ = rec_m["rew"][:, :, 1].sum(0).cpu().numpy()
    b_ = rec["rew"][:, :, 1].sum(0).cpu().numpy()
    rng2 = np.random.default_rng(1)
    bs = (a_[rng2.integers(0, len(a_), (4000, len(a_)))].mean(1)
          - b_[rng2.integers(0, len(b_), (4000, len(b_)))].mean(1))
    out["nie"] = nie
    out["nie_lo"], out["nie_hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    out["nie_arrivals"] = float(rec_m["arrived"][:, :, 1].float().sum(0).mean())
    del tr_m, env_m
    torch.cuda.empty_cache()

    # --- payoff consequence of deleting the subtype ------------------------
    res = {}
    res["intact"], _ = evaluate_channel(tr, env, None, steps, seed, target=0)
    env._kill_subtype = True
    res["kill"], _ = evaluate_channel(tr, env, None, steps, seed, target=0)
    env._kill_subtype = False
    env._scramble_subtype = True
    res["scramble"], _ = evaluate_channel(tr, env, None, steps, seed, target=0)
    env._scramble_subtype = False
    for k in ("kill", "scramble"):
        pdv = paired_diff(res, k, "intact", "reward_others")
        out[f"{k}_d_receiver"] = pdv["mean"]
        out[f"{k}_lo"], out[f"{k}_hi"] = pdv["lo"], pdv["hi"]
        out[f"{k}_p"] = pdv["p_two_sided"]
    del tr, env
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    rows = []
    for script in ("honest", "random"):
        for listen in (True, False):
            r = run_condition(script, listen)
            rows.append(r)
            print(f"[{script:6s} listen={str(listen):5s}] "
                  f"arrivals={r['arrivals_per_ep']:.2f}/4 return={r['receiver_return']:6.1f} "
                  f"MI={r['content_mi']:.4f}(null {r['content_null']:.4f}) "
                  f"CIE={r['cie']:.4f}(null {r['cie_null']:.5f}) PL={r['pl_shift']:.4f} "
                  f"kill_dR={r['kill_d_receiver']:+.2f}  NIE={r['nie']:+.2f} "
                  f"[{r['nie_lo']:+.2f},{r['nie_hi']:+.2f}]",
                  flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/assay_validation.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote results/assay_validation.json")
