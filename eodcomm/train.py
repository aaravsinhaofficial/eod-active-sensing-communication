"""Training driver: one condition, one seed."""

from __future__ import annotations

import json
import os
import time

import torch

from .env import EnvConfig, FishEnv
from .ppo import MAPPO, PPOConfig


def train_one(
    env_kwargs: dict,
    seed: int = 0,
    total_steps: int = 30_000_000,
    ppo_kwargs: dict | None = None,
    device: str = "cuda",
    log_every: int = 8,
    out_dir: str | None = None,
    tag: str = "run",
    verbose: bool = True,
):
    ecfg = EnvConfig(device=device, **env_kwargs)
    pcfg = PPOConfig(**(ppo_kwargs or {}))
    env = FishEnv(ecfg)
    obs = env.reset(seed)
    tr = MAPPO(env, pcfg, device=device, seed=seed)

    per_iter = pcfg.rollout * env.B * env.F
    n_iter = max(1, total_steps // per_iter)
    iters_per_ep = max(1, ecfg.episode_len // pcfg.rollout)

    hist = []
    acc = {"ret": 0.0, "eat": 0.0, "emit": 0.0, "struck": 0.0, "n": 0}
    t0 = time.time()
    for it in range(n_iter):
        obs, buf, lv, infos = tr.rollout(obs, record={"ate", "struck", "emit_self"})
        tr.update(buf, lv)
        acc["ret"] += torch.stack(buf["rew"]).sum(0).mean().item()
        acc["eat"] += torch.stack([i["ate"] for i in infos]).sum(0).mean().item()
        acc["emit"] += torch.stack([i["emit_self"] for i in infos]).to(torch.float32).mean().item()
        acc["struck"] += torch.stack([i["struck"] for i in infos]).sum(0).mean().item()
        acc["n"] += 1
        if (it + 1) % (log_every * iters_per_ep) == 0 or it == n_iter - 1:
            k = acc["n"] / iters_per_ep
            rec = {
                "iter": it + 1,
                "steps": (it + 1) * per_iter,
                "return_per_episode": acc["ret"] / k,
                "eaten_per_episode": acc["eat"] / k,
                "emit_rate": acc["emit"] / acc["n"],
                "struck_per_episode": acc["struck"] / k,
            }
            hist.append(rec)
            if verbose:
                print(
                    f"[{tag}] {rec['steps']/1e6:5.1f}M  return={rec['return_per_episode']:+8.2f} "
                    f"eaten={rec['eaten_per_episode']:5.2f} emit={rec['emit_rate']:.3f} "
                    f"struck={rec['struck_per_episode']:.2f}",
                    flush=True,
                )
            acc = {"ret": 0.0, "eat": 0.0, "emit": 0.0, "struck": 0.0, "n": 0}

    wall = time.time() - t0
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {"net": tr.net.state_dict(), "env_kwargs": env_kwargs, "ppo_kwargs": ppo_kwargs or {}, "seed": seed},
            os.path.join(out_dir, f"{tag}.pt"),
        )
        with open(os.path.join(out_dir, f"{tag}_hist.json"), "w") as f:
            json.dump({"hist": hist, "wall_s": wall, "env_kwargs": {k: str(v) for k, v in env_kwargs.items()}}, f, indent=2)
    return tr, env, hist


def load_agent(path: str, device: str = "cuda", env_overrides: dict | None = None):
    ck = torch.load(path, map_location=device, weights_only=False)
    ek = dict(ck["env_kwargs"])
    ek.update(env_overrides or {})
    env = FishEnv(EnvConfig(device=device, **ek))
    tr = MAPPO(env, PPOConfig(**ck["ppo_kwargs"]), device=device, seed=ck["seed"])
    tr.net.load_state_dict(ck["net"])
    tr.net.eval()
    return tr, env, ck
