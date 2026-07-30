"""A dyad that is known to be communicating, used to validate the measurements.

Every negative result in this paper rests on an assay saying "no communication
here". An assay that always says that would produce the same result, so the
assays have to be shown to fire on a case where communication is not in doubt.

Here the sender's discharge subtype *is* the private cue (the environment
scripts it), and the receiver is a fixed reactive controller that steers toward
whichever site the subtype names. Communication exists by construction. The
controller is a function of the observation alone -- it reads the subtype and
the egocentric site bearings out of the observation vector, never out of the
simulator state -- so every intervention that re-renders an observation
propagates to its behaviour exactly as it would for a learned policy, and the
whole metric battery applies unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ScriptedReferentialNet(nn.Module):
    """Drop-in replacement for `ActorCritic` with the same `.policy()` contract.

    Observation layout in referential mode, counting from the end:
        -6  heard subtype from the single conspecific
        -5  own private cue (non-zero for the sender only)
        -4  cos(bearing to site 0)
        -3  cos(bearing to site 1)
        -2  sin(bearing to site 0)
        -1  sin(bearing to site 1)
    """

    def __init__(self, obs_dim: int, n_agents: int, cfg, n_disc: int = 3,
                 listen: bool = True, turn_gain: float = 6.0, log_std: float = -2.0):
        super().__init__()
        self.n_disc = n_disc
        self.gru = type("g", (), {"hidden_size": cfg.gru})()
        self.listen = listen
        self.turn_gain = turn_gain
        self._log_std = log_std
        # a parameter so .to(device) and .state_dict() behave normally
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def policy(self, obs, hx):
        sub = obs[..., -6]                       # heard subtype, in {-1, 0, +1}
        cos0, cos1 = obs[..., -4], obs[..., -3]
        sin0, sin1 = obs[..., -2], obs[..., -1]

        # The environment encodes site index i as subtype 2i-1, so -1 names
        # site 0 and +1 names site 1; with no signal, split the difference.
        w0 = (sub < 0).to(obs.dtype)
        w1 = (sub > 0).to(obs.dtype)
        if not self.listen:                      # deaf control: ignore the subtype
            w0 = torch.full_like(w0, 0.5)
            w1 = torch.full_like(w1, 0.5)
        none = 1.0 - w0 - w1
        cos_t = w0 * cos0 + w1 * cos1 + none * 0.5 * (cos0 + cos1)
        sin_t = w0 * sin0 + w1 * sin1 + none * 0.5 * (sin0 + sin1)

        bearing = torch.atan2(sin_t, cos_t)      # egocentric angle to the target
        turn = torch.tanh(self.turn_gain * bearing)
        thrust = torch.clamp(cos_t, min=0.0) * 0.0 + 1.0   # always drive forward
        mu = torch.stack([thrust, turn], dim=-1).clamp(-1, 1)
        ls = torch.full_like(mu, self._log_std)
        # emit always (the sender's schedule is forced by the env anyway); never
        # bite; subtype logit is irrelevant for the receiver and overridden for
        # the sender
        logit = torch.zeros(mu.shape[:-1] + (self.n_disc,), device=obs.device, dtype=obs.dtype)
        logit[..., 0] = 4.0
        logit[..., 1] = -4.0
        return mu, ls, logit, hx

    def value(self, cobs, hx):
        return torch.zeros(cobs.shape[:-1], device=cobs.device, dtype=cobs.dtype), hx


class ScriptedRunner:
    """Minimal stand-in for `MAPPO` so the metric functions accept it unchanged."""

    def __init__(self, env, net, device="cuda"):
        self.env, self.net, self.dev = env, net.to(device), torch.device(device)
        self.B, self.F, self.D = env.B, env.F, env.obs_dim
        self.N = self.B * self.F
        self.cfg = type("c", (), {"gru": 8})()
        self.reset_state()

    def reset_state(self):
        self.hx = torch.zeros((1, self.N, 8), device=self.dev)
        self.chx = torch.zeros((1, self.B, 8), device=self.dev)
