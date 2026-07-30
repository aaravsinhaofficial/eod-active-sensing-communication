"""A minimal dual-use foraging world, with no electric fish in it.

The claim the electric-fish results suggest is not about electroreception. It is
about any action that senses and broadcasts at once: an active sonar ping, a
robot's laser sweep, a rustle in undergrowth, a search query in a shared log.
This environment strips that structure down to its essentials so the same audit
can be run somewhere the biology is absent.

An agent's PING has exactly the three consequences the discharge has:

  reafference   the pinger learns where nearby items are
  illumination  agents close to the pinger learn where items near *them* are,
                for free, because the ping lit up their neighbourhood too
  detection     every agent learns that this particular agent pinged, and
                roughly from where

The three are independently maskable through the same interface as the fish
environment, so `MAPPO`, the metric battery and the intervention battery all run
against it unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class DualUseConfig:
    n_agents: int = 4
    n_items: int = 40
    arena: float = 1.0
    episode_len: int = 512
    batch: int = 512
    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    n_bins: int = 8            # bearing histogram resolution
    r_self: float = 0.22       # how far a ping shows the pinger
    r_illum: float = 0.18      # how close you must be to benefit from another's ping
    r_detect: float = 0.75     # how far a ping is detectable as an event
    r_passive: float = 0.10    # a weak always-on sense, so that silencing an
                               # agent is not the same as blinding it outright
    r_collect: float = 0.035
    speed: float = 0.012

    ping_cost: float = 0.0
    shared_reward: float = 0.0
    r_item: float = 1.0

    # the same three masks as the fish environment
    reafference: bool = True
    illuminate_others: bool = True
    detection: bool = True
    ping_allowed: bool = True
    signal_channel: bool = False

    noise: float = 0.05


class DualUseEnv:
    def __init__(self, cfg: DualUseConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.dt_type = cfg.dtype
        self.B, self.F = cfg.batch, cfg.n_agents
        self.n_slots = self.F - 1
        self.obs_dim = (
            cfg.n_bins           # own reafferent map
            + cfg.n_bins         # map borrowed from neighbours' pings
            + cfg.n_bins         # weak passive map, always on
            + self.n_slots * 3   # per-emitter: heard, cos, sin of bearing to it
            + 2                  # own position
            + 3                  # last action
            + (self.n_slots if cfg.signal_channel else 0)
        )
        self.n_disc = 2 if cfg.signal_channel else 1   # ping [, signal]
        self.act_dim = 2 + self.n_disc
        self.g = torch.Generator(device=self.dev)
        self.g.manual_seed(0)
        self._odi = torch.tensor(
            [[j for j in range(self.F) if j != i] for i in range(self.F)],
            device=self.dev, dtype=torch.long)

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        cfg = self.cfg
        if seed is not None:
            self.g.manual_seed(seed)
        B, F = self.B, self.F
        self.pos = torch.rand((B, F, 2), generator=self.g, device=self.dev, dtype=self.dt_type)
        self.items = torch.rand((B, cfg.n_items, 2), generator=self.g, device=self.dev, dtype=self.dt_type)
        self.alive = torch.ones((B, cfg.n_items), device=self.dev, dtype=torch.bool)
        self.last_act = torch.zeros((B, F, 3), device=self.dev, dtype=self.dt_type)
        self._sig = torch.zeros((B, F), device=self.dev, dtype=torch.bool)
        # the audit pipeline expects a heading and a food array; this world has
        # neither, so expose the nearest equivalents under the same names
        self.theta = torch.zeros((B, F), device=self.dev, dtype=self.dt_type)
        self.size = torch.full((B, F), 0.5, device=self.dev, dtype=self.dt_type)
        self.t = 0
        z = torch.zeros((B, F), device=self.dev, dtype=torch.bool)
        return self._observe(z, z, None, z)

    # ------------------------------------------------------------------
    def _bearing_map(self, centre, ping_mask, radius):
        """Histogram of item bearings around `centre`, gated by `ping_mask`."""
        cfg = self.cfg
        B, F = self.B, self.F
        rel = self.items[:, None, :, :] - centre[:, :, None, :]        # (B,F,N,2)
        d = rel.norm(dim=-1)
        vis = (d < radius) & self.alive[:, None, :]
        ang = torch.atan2(rel[..., 1], rel[..., 0])
        b = ((ang + math.pi) / (2 * math.pi) * cfg.n_bins).long().clamp(0, cfg.n_bins - 1)
        out = torch.zeros((B, F, cfg.n_bins), device=self.dev, dtype=self.dt_type)
        w = torch.where(vis, 1.0 - d / radius, torch.zeros_like(d))
        out.scatter_add_(2, b, w)
        return out * ping_mask[..., None].to(self.dt_type)

    def _observe(self, ping_self, ping_det=None, slot_perm=None, ping_illum=None):
        if ping_det is None:
            ping_det = ping_self
        if ping_illum is None:
            ping_illum = ping_self if self.cfg.illuminate_others else torch.zeros_like(ping_self)
        cfg = self.cfg
        B, F = self.B, self.F

        # (1) reafference: my own ping shows me my own neighbourhood
        own = self._bearing_map(self.pos, ping_self, cfg.r_self)
        if not cfg.reafference:
            own = torch.zeros_like(own)

        # (2) illumination: a neighbour's ping shows me *my* neighbourhood, if it
        # was close enough for its ping to reach my surroundings
        d_ag = torch.cdist(self.pos, self.pos)
        eye = torch.eye(F, device=self.dev, dtype=torch.bool)[None]
        lit = ((d_ag < cfg.r_illum) & ~eye & ping_illum[:, None, :]).any(-1)
        borrowed = self._bearing_map(self.pos, lit, cfg.r_self) * 0.6

        # (3) detection: who pinged, and roughly where they are
        idx = self._odi
        heard = (ping_det[:, None, :].expand(B, F, F) & (d_ag < cfg.r_detect) & ~eye)
        heard = torch.gather(heard, 2, idx[None].expand(B, F, F - 1)).to(self.dt_type)
        rel = self.pos[:, None, :, :] - self.pos[:, :, None, :]
        ang = torch.atan2(rel[..., 1], rel[..., 0])
        ang = torch.gather(ang, 2, idx[None].expand(B, F, F - 1))
        if slot_perm is not None:
            heard = torch.gather(heard, 2, slot_perm)
            ang = torch.gather(ang, 2, slot_perm)
        if not cfg.detection:
            heard = torch.zeros_like(heard)
        det = torch.cat([heard, torch.cos(ang) * heard, torch.sin(ang) * heard], -1)

        extra = []
        if cfg.signal_channel:
            sg = (self._sig.to(self.dt_type) * 2 - 1)[:, None, :].expand(B, F, F)
            sg = torch.gather(sg, 2, idx[None].expand(B, F, F - 1))
            if slot_perm is not None:
                sg = torch.gather(sg, 2, slot_perm)
            if getattr(self, "_kill_subtype", False):
                sg = torch.zeros_like(sg)
            extra.append(sg * heard)

        n = cfg.noise
        ones = torch.ones((B, F), device=self.dev, dtype=torch.bool)
        passive = self._bearing_map(self.pos, ones, cfg.r_passive) * 0.5
        obs = torch.cat([own, borrowed, passive, det, self.pos, self.last_act] + extra, -1)
        obs = obs * (1 + (torch.rand(obs.shape, generator=self.g, device=self.dev,
                                     dtype=self.dt_type) * 2 - 1) * n)
        return obs

    # ------------------------------------------------------------------
    def _apply_channel(self, ping, spec):
        illum = ping if self.cfg.illuminate_others else torch.zeros_like(ping)
        if spec is None or spec.mode == "intact":
            return ping, illum, ping, None
        sel = torch.zeros((self.F,), device=self.dev, dtype=torch.bool)
        sel[:] = True if spec.agents is None else False
        if spec.agents is not None:
            sel[list(spec.agents)] = True
        sel = sel[None].expand(self.B, self.F)
        s, i_, d, perm = ping, illum, ping, None
        m = spec.mode
        if m == "mute":
            s, i_, d = ping & ~sel, illum & ~sel, ping & ~sel
        elif m == "private":
            i_, d = illum & ~sel, ping & ~sel
        elif m == "cue_only":
            d = ping & ~sel
        elif m == "signal_only":
            i_ = illum & ~sel
        elif m == "social":
            s = ping & ~sel
        elif m == "phantom":
            ph = torch.rand((self.B, self.F), generator=self.g, device=self.dev) < spec.phantom_rate
            d = torch.where(sel, ph, ping)
        elif m in ("replay", "scramble_time"):
            tr = spec.replay_train
            d = torch.where(sel, tr[:, self.t % tr.shape[1]].to(self.dev), ping)
        elif m == "scramble_id":
            perm = torch.argsort(torch.rand((self.B, self.F, self.F - 1),
                                            generator=self.g, device=self.dev), dim=-1)
        return s, i_, d, perm

    def step(self, action, channel=None):
        cfg = self.cfg
        B, F = self.B, self.F
        action = action.clamp(-1, 1)
        ping = (action[..., 2] > 0) & cfg.ping_allowed
        self._sig = ((action[..., 3] > 0) & ping) if cfg.signal_channel else torch.zeros_like(ping)
        p_self, p_illum, p_det, perm = self._apply_channel(ping, channel)

        self.pos = (self.pos + action[..., :2] * cfg.speed).clamp(0.0, cfg.arena)

        d = torch.cdist(self.pos, self.items)
        got = (d < cfg.r_collect) & self.alive[:, None, :]
        first = got & (got.cumsum(1) == 1)
        n_got = first.sum(-1).to(self.dt_type)
        self.alive = self.alive & ~first.any(1)
        # respawn so the task is stationary
        resp = ~self.alive
        newp = torch.rand(self.items.shape, generator=self.g, device=self.dev, dtype=self.dt_type)
        self.items = torch.where(resp[..., None], newp, self.items)
        self.alive = self.alive | resp

        r = cfg.r_item * n_got
        if cfg.shared_reward > 0:
            grp = n_got.sum(1, keepdim=True) - n_got
            r = r + cfg.r_item * cfg.shared_reward * grp / max(F - 1, 1)
        r = r - cfg.ping_cost * p_self.to(self.dt_type)

        self.last_act = action[..., :3]
        self.theta = torch.atan2(action[..., 1], action[..., 0])
        self.t += 1
        obs = self._observe(p_self, p_det, perm, p_illum)
        info = {"emit": ping, "emit_self": p_self, "emit_social": p_det,
                "emit_illum": p_illum, "signal": self._sig, "ate": n_got,
                "struck": torch.zeros_like(n_got), "bit": torch.zeros_like(n_got),
                "bitten": torch.zeros_like(n_got),
                "heard": torch.zeros((B, F, F - 1), device=self.dev, dtype=self.dt_type),
                "nearest_food": d.min(-1).values}
        self.food = self.items
        self.food_alive = self.alive
        self.pred_pos = None
        return obs, r, self.t >= cfg.episode_len, info
