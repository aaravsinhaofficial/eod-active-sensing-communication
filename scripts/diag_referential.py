"""Why can a hand-coded controller solve the referential game when PPO cannot?

The scripted receiver reaches the correct site on every trial from exactly the
observation the learned one gets, so the information is present and a policy
exists.  This script varies the two things most likely to be responsible: the
magnitude of the exploration noise on the steering action, and the length of the
credit-assignment horizon between committing to a direction and arriving.
"""

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from eodcomm.env import EnvConfig, FishEnv  # noqa: E402
from eodcomm.ppo import MAPPO, PPOConfig  # noqa: E402

BASE = dict(n_fish=2, n_food=10, task="referential", signal_channel=True,
            arena_cm=(90.0, 70.0), shared_food=1.0, episode_len=512,
            size_mode="none", ref_scripted="honest", batch=512)
NEAR = dict(ref_sites=((25.0, 35.0), (65.0, 35.0)), ref_start_xy=(45.0, 16.0),
            ref_sender_xy=(45.0, 8.0), ref_trial_len=80)

CFGS = [
    ("far  noise-lo", {}, -2.0),
    ("near noise-hi", NEAR, -0.5),
    ("near noise-lo", NEAR, -2.0),
]

if __name__ == "__main__":
    for name, over, ls in CFGS:
        res = []
        for seed in (0, 1):
            cfg = EnvConfig(**{**BASE, **over})
            env = FishEnv(cfg)
            obs = env.reset(seed)
            tr = MAPPO(env, PPOConfig(rollout=64, ent_coef=0.02, log_std_init=ls), seed=seed)
            arr = 0.0
            for it in range(160):
                obs, buf, lv, infos = tr.rollout(obs, record={"arrived"})
                tr.update(buf, lv)
                if it >= 152:
                    arr += torch.stack([i["arrived"] for i in infos])[:, :, 1].float().sum(0).mean().item()
            res.append(arr)
            del tr, env
            torch.cuda.empty_cache()
        ntr = 512 // over.get("ref_trial_len", 128)
        print(f"{name:14s} arrivals/ep={np.mean(res):5.2f} of {ntr} trials "
              f"(chance {ntr/2:.1f})  seeds={[round(b, 2) for b in res]}", flush=True)
