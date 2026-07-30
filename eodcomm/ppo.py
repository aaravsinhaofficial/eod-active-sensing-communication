"""Recurrent multi-agent PPO with a centralised critic.

Agents share one set of actor parameters (standard MAPPO practice) and are
distinguished by their observations, which include body size -- the persistent
dominance cue.  The action space is hybrid: two squashed Gaussians for
locomotion and two Bernoullis for the discharge and bite decisions.  Keeping the
discharge decision explicitly Bernoulli matters for the analysis, because the
emission probability is exactly the quantity we intervene on when computing
counterfactual influence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    hidden: int = 128
    gru: int = 128
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    ent_coef: float = 0.01
    ent_coef_emit: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout: int = 64
    log_std_init: float = -0.5
    # Eccles et al. (2019) positive-signalling bias.  Emergent communication has
    # a chicken-and-egg problem: a sender has no gradient toward an informative
    # message until a receiver attends to it, and a receiver has none toward
    # attending until the message is informative.  This auxiliary loss pushes the
    # message head to be confident given the state but varied across states,
    # which breaks the deadlock from the sender's side without telling it what to
    # say.
    ps_bias: float = 0.0
    ps_target_entropy: float = 0.5


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_agents: int, cfg: PPOConfig, n_disc: int = 2):
        super().__init__()
        self.n_disc = n_disc
        h, g = cfg.hidden, cfg.gru
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, h), nn.LayerNorm(h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
        )
        self.gru = nn.GRU(h, g, batch_first=False)
        self.pi_cont = nn.Linear(g, 2)      # thrust, turn (means)
        self.pi_disc = nn.Linear(g, n_disc)  # emit, bite [, signal] (logits)
        self.log_std = nn.Parameter(torch.full((2,), cfg.log_std_init))

        self.cenc = nn.Sequential(
            nn.Linear(obs_dim * n_agents, h), nn.LayerNorm(h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
        )
        self.cgru = nn.GRU(h, g, batch_first=False)
        self.v = nn.Linear(g, 1)

        for m in (self.pi_cont, self.pi_disc, self.v):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    # -- actor -----------------------------------------------------------
    def policy(self, obs, hx):
        """obs: (T, N, D), hx: (1, N, G). Returns distribution params and new state."""
        z = self.enc(obs)
        z, hx = self.gru(z, hx)
        mu = torch.tanh(self.pi_cont(z))
        logit = self.pi_disc(z)
        return mu, self.log_std.expand_as(mu), logit, hx

    def value(self, cobs, hx):
        z = self.cenc(cobs)
        z, hx = self.cgru(z, hx)
        return self.v(z).squeeze(-1), hx


# ---------------------------------------------------------------------------
# Hybrid action distribution helpers
# ---------------------------------------------------------------------------

def sample_action(mu, log_std, logit, generator=None):
    std = log_std.exp()
    eps = torch.randn(mu.shape, device=mu.device, generator=generator)
    cont = (mu + std * eps).clamp(-1, 1)
    p = torch.sigmoid(logit)
    u = torch.rand(p.shape, device=p.device, generator=generator)
    disc = (u < p).to(mu.dtype)
    act = torch.cat([cont, disc * 2 - 1], dim=-1)  # env thresholds at 0
    return act, cont, disc


def log_prob(mu, log_std, logit, cont, disc):
    std = log_std.exp()
    lp_c = (-0.5 * ((cont - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
    lp_d = -F.binary_cross_entropy_with_logits(logit, disc, reduction="none").sum(-1)
    return lp_c + lp_d


def entropy(log_std, logit):
    ent_c = (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)
    p = torch.sigmoid(logit)
    ent_d = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
    return ent_c + ent_d.sum(-1), ent_d


def kl_hybrid(mu1, ls1, lg1, mu2, ls2, lg2):
    """Analytic KL( pi_1 || pi_2 ) for the factored Gaussian x Bernoulli policy."""
    v1, v2 = (2 * ls1).exp(), (2 * ls2).exp()
    kl_c = (ls2 - ls1 + (v1 + (mu1 - mu2) ** 2) / (2 * v2) - 0.5).sum(-1)
    p1, p2 = torch.sigmoid(lg1), torch.sigmoid(lg2)
    eps = 1e-8
    kl_d = (
        p1 * (torch.log(p1 + eps) - torch.log(p2 + eps))
        + (1 - p1) * (torch.log(1 - p1 + eps) - torch.log(1 - p2 + eps))
    ).sum(-1)
    return kl_c + kl_d


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MAPPO:
    def __init__(self, env, cfg: PPOConfig, device="cuda", seed=0):
        self.env, self.cfg, self.dev = env, cfg, torch.device(device)
        torch.manual_seed(seed)
        self.net = ActorCritic(env.obs_dim, env.F, cfg,
                               n_disc=getattr(env, "n_disc", 2)).to(self.dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.B, self.F, self.D = env.B, env.F, env.obs_dim
        self.N = self.B * self.F
        self.reset_state()

    def reset_state(self):
        g = self.cfg.gru
        self.hx = torch.zeros((1, self.N, g), device=self.dev)
        self.chx = torch.zeros((1, self.B, g), device=self.dev)

    # ------------------------------------------------------------------
    def _central(self, obs):
        return obs.reshape(self.B, self.F * self.D)

    @torch.no_grad()
    def rollout(self, obs, channel=None, record=None):
        cfg, T = self.cfg, self.cfg.rollout
        buf = {k: [] for k in ("obs", "cobs", "cont", "disc", "logp", "val", "rew", "hx", "chx", "done")}
        infos = []
        for _ in range(T):
            flat = obs.reshape(1, self.N, self.D)
            buf["hx"].append(self.hx.clone())
            buf["chx"].append(self.chx.clone())
            mu, ls, lg, self.hx = self.net.policy(flat, self.hx)
            act, cont, disc = sample_action(mu[0], ls[0], lg[0])
            lp = log_prob(mu[0], ls[0], lg[0], cont, disc)
            cobs = self._central(obs)
            val, self.chx = self.net.value(cobs[None], self.chx)

            nobs, rew, done, info = self.env.step(
                act.reshape(self.B, self.F, self.env.act_dim), channel)
            buf["obs"].append(obs)
            buf["cobs"].append(cobs)
            buf["cont"].append(cont)
            buf["disc"].append(disc)
            buf["logp"].append(lp)
            buf["val"].append(val[0])
            buf["rew"].append(rew.reshape(self.N))
            buf["done"].append(float(done))
            if record is not None:
                infos.append({k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in info.items() if k in record})
            obs = nobs
            if done:
                obs = self.env.reset()
                self.reset_state()
        with torch.no_grad():
            cobs = self._central(obs)
            last_val, _ = self.net.value(cobs[None], self.chx)
        return obs, buf, last_val[0], infos

    def update(self, buf, last_val):
        cfg = self.cfg
        T = len(buf["rew"])
        rew = torch.stack(buf["rew"])                     # (T, N)
        val = torch.stack(buf["val"])                     # (T, B)
        val_n = val[:, :, None].expand(T, self.B, self.F).reshape(T, self.N)
        last_n = last_val[:, None].expand(self.B, self.F).reshape(self.N)

        done = torch.tensor(buf["done"], device=rew.device, dtype=rew.dtype)
        adv = torch.zeros_like(rew)
        gae = torch.zeros_like(rew[0])
        nxt = last_n
        for t in reversed(range(T)):
            nonterm = 1.0 - done[t]
            delta = rew[t] + cfg.gamma * nxt * nonterm - val_n[t]
            gae = delta + cfg.gamma * cfg.lam * gae * nonterm
            adv[t] = gae
            nxt = val_n[t]
        ret = adv + val_n
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.stack(buf["obs"]).reshape(T, self.N, self.D)
        cobs = torch.stack(buf["cobs"])
        cont = torch.stack(buf["cont"])
        disc = torch.stack(buf["disc"])
        logp_old = torch.stack(buf["logp"])
        hx0 = buf["hx"][0]
        chx0 = buf["chx"][0]

        stats = {}
        nmb = cfg.minibatches
        env_chunk = self.B // nmb
        for _ in range(cfg.epochs):
            perm = torch.randperm(self.B, device=self.dev)
            for i in range(nmb):
                eidx = perm[i * env_chunk:(i + 1) * env_chunk]
                aidx = (eidx[:, None] * self.F + torch.arange(self.F, device=self.dev)[None]).reshape(-1)

                mu, ls, lg, _ = self.net.policy(obs[:, aidx], hx0[:, aidx])
                lp = log_prob(mu, ls, lg, cont[:, aidx], disc[:, aidx])
                ratio = (lp - logp_old[:, aidx]).exp()
                a = adv[:, aidx]
                l1 = ratio * a
                l2 = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a
                pl = -torch.min(l1, l2).mean()

                v, _ = self.net.value(cobs[:, eidx], chx0[:, eidx])
                v_n = v[:, :, None].expand(T, len(eidx), self.F).reshape(T, -1)
                vl = F.mse_loss(v_n, ret[:, aidx])

                ent_all, ent_d = entropy(ls, lg)
                ps_loss = torch.zeros((), device=lg.device)
                if cfg.ps_bias > 0 and lg.shape[-1] >= 3:
                    # index 2 is the discharge-subtype (message) head
                    p_msg = torch.sigmoid(lg[..., 2])
                    pbar = p_msg.mean()
                    H_avg = -(pbar * torch.log(pbar + 1e-8)
                              + (1 - pbar) * torch.log(1 - pbar + 1e-8))
                    H_cond = ent_d[..., 2].mean()
                    # want high marginal entropy, low conditional entropy
                    ps_loss = -(H_avg - H_cond)
                # the discharge decision gets its own exploration bonus: without it
                # the emit head collapses long before foraging is learned
                ent = ent_all.mean() + cfg.ent_coef_emit * ent_d[..., 0].mean()

                loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent + cfg.ps_bias * ps_loss
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                stats = {"pi_loss": pl.item(), "v_loss": vl.item(), "entropy": ent.item()}
        return stats
