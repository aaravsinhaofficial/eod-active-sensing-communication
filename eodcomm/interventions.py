"""Counterfactual channel experiments run on frozen, trained policies.

The central object is a *matched pair* of rollouts that share initial
conditions and the sensory-noise stream and differ only in how one agent's
discharges are routed.  Because the private (reafferent) and public
(conspecific-audible) consequences of a pulse can be switched independently,
the effect of "silencing a fish" -- a single, confounded manipulation upstream
-- decomposes into an additive pair of effects that can be attributed
separately.
"""

from __future__ import annotations

import numpy as np
import torch

from .env import ChannelSpec
from .metrics import collect


# ---------------------------------------------------------------------------
# Building surrogate pulse trains
# ---------------------------------------------------------------------------

@torch.no_grad()
def record_train(tr, env, n_steps: int, seed: int = 123):
    """Record the factual emission train, shape (B, T, F)."""
    rec = collect(tr, env, n_steps, seed=seed, deterministic=False)
    return rec["emit"].permute(1, 0, 2).bool().contiguous(), rec


def cross_arena_replay(train: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Replay each agent's train from a *different arena*.

    Marginal rate, burstiness and inter-pulse-interval statistics are exactly
    preserved -- these are the same trains the same policy produced -- but the
    contingency between a pulse and the receiver's current world is destroyed.
    This is the playback experiment of the electric-fish literature, done with
    perfect stimulus matching.
    """
    g = torch.Generator(device=train.device).manual_seed(seed)
    perm = torch.randperm(train.shape[0], generator=g, device=train.device)
    return train[perm].clone()


def time_scramble(train: torch.Tensor, seed: int = 0, min_shift: int = 166) -> torch.Tensor:
    """Circularly shift each (arena, agent) train, preserving rate, destroying timing."""
    g = torch.Generator(device=train.device).manual_seed(seed)
    B, T, F = train.shape
    ms = max(1, min(min_shift, T // 4))
    sh = torch.randint(ms, T - ms, (B, F), generator=g, device=train.device)
    idx = (torch.arange(T, device=train.device)[None, :, None] + sh[:, None, :]) % T
    out = torch.gather(train, 1, idx)
    return out


def context_swap_replay(train: torch.Tensor, context: torch.Tensor, seed: int = 0):
    """Replay trains recorded in the opposite behavioural context.

    `context` is (B, T, F) boolean -- here, whether the emitter had food within
    10 cm.  For each agent we pair each arena with another arena whose context
    time-series is maximally anti-correlated, so a receiver that responds to
    pulse *content* rather than pulse *presence* should be misled.
    """
    g = torch.Generator(device=train.device).manual_seed(seed)
    B = train.shape[0]
    c = context.float().mean(1)                      # (B,F) fraction of time near food
    order = torch.argsort(c.mean(1))
    partner = torch.empty(B, dtype=torch.long, device=train.device)
    partner[order] = order.flip(0)                   # rich <-> poor
    return train[partner].clone()


# ---------------------------------------------------------------------------
# Matched-pair evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_channel(tr, env, channel: ChannelSpec | None, n_steps: int, seed: int,
                     target: int = 0):
    """Run one condition and summarise the behaviour of target and non-targets."""
    rec = collect(tr, env, n_steps, channel=channel, seed=seed, deterministic=False)
    F = env.F
    others = [j for j in range(F) if j != target]
    pos = rec["pos"]
    d = torch.cdist(pos, pos)                                # (T,B,F,F)
    eye = torch.eye(F, device=d.device, dtype=torch.bool)
    nn_dist = d.masked_fill(eye[None, None], float("inf")).min(-1).values

    # how strongly do non-targets orient toward the target?
    to_t = pos[:, :, target:target + 1, :] - pos
    bearing = torch.atan2(to_t[..., 1], to_t[..., 0]) - rec["theta"]
    bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))
    facing = torch.cos(bearing)                              # 1 = heading at target

    # per-arena totals, kept so that every condition comparison can be made
    # paired within arena (same initial layout, same noise stream)
    per_arena = {
        "eaten_target": rec["ate"][:, :, target].sum(0),
        "eaten_others": rec["ate"][:, :, others].sum(0).mean(-1),
        "eaten_group": rec["ate"].sum(0).sum(-1),
        "reward_target": rec["rew"][:, :, target].sum(0),
        "reward_others": rec["rew"][:, :, others].sum(0).mean(-1),
        "nn_dist": nn_dist.mean(0).mean(-1),
    }
    out = {
        "eaten_target": rec["ate"][:, :, target].sum(0).mean().item(),
        "eaten_others": rec["ate"][:, :, others].sum(0).mean().item(),
        "eaten_group": rec["ate"].sum(0).sum(-1).mean().item(),
        "reward_target": rec["rew"][:, :, target].sum(0).mean().item(),
        "reward_others": rec["rew"][:, :, others].sum(0).mean().item(),
        "emit_rate": rec["emit"].mean().item(),
        "emit_rate_target": rec["emit"][:, :, target].mean().item(),
        "nn_dist": nn_dist.mean().item(),
        "nn_dist_target": nn_dist[:, :, target].mean().item(),
        "facing_target": facing[:, :, others].mean().item(),
        "dist_to_target": d[:, :, target, others].mean().item(),
        "struck": rec["struck"].sum(0).mean().item(),
        "bit": rec["bit"].sum(0).mean().item(),
        "heard_frac": rec["heard"].mean().item(),
    }
    out["_per_arena"] = {k: v.detach().cpu().numpy() for k, v in per_arena.items()}
    return out, rec


def channel_battery(tr, env, n_steps: int = 512, seed: int = 7, target: int = 0,
                    phantom_rates=(0.05, 0.2, 0.6)):
    """The full set of counterfactual channels, all evaluated on one frozen policy."""
    train, rec0 = record_train(tr, env, n_steps, seed=seed)
    ctx = (rec0["food_near10"] > 0).permute(1, 0, 2).contiguous()

    specs = {
        "intact": None,
        "mute_both": ChannelSpec(mode="mute", agents=(target,)),
        "mute_self": ChannelSpec(mode="social", agents=(target,)),     # keeps public reach
        "mute_social": ChannelSpec(mode="private", agents=(target,)),  # keeps reafference
        "mute_signal": ChannelSpec(mode="cue_only", agents=(target,)),  # illuminates but undetected
        "mute_cue": ChannelSpec(mode="signal_only", agents=(target,)),  # detected but no illumination
        "replay_cross": ChannelSpec(mode="replay", agents=(target,),
                                    replay_train=cross_arena_replay(train, seed)),
        "replay_context": ChannelSpec(mode="replay", agents=(target,),
                                      replay_train=context_swap_replay(train, ctx, seed)),
        "scramble_time": ChannelSpec(mode="scramble_time", agents=(target,),
                                     replay_train=time_scramble(train, seed)),
        "scramble_id": ChannelSpec(mode="scramble_id"),
    }
    for r in phantom_rates:
        specs[f"phantom_{r}"] = ChannelSpec(mode="phantom", agents=(target,), phantom_rate=r)

    res = {}
    for k, sp in specs.items():
        res[k], _ = evaluate_channel(tr, env, sp, n_steps, seed, target)

    # If a decoupled signalling variable exists, ablate it on its own.  The pulse
    # -- and therefore every sensory consequence -- is left completely intact, so
    # any payoff change is attributable to the signal content alone.  This is the
    # manipulation that is impossible for the coupled channel.
    if getattr(env.cfg, "signal_channel", False):
        for flag, name in (("_kill_subtype", "kill_subtype"),
                           ("_scramble_subtype", "scramble_subtype")):
            setattr(env, flag, True)
            res[name], _ = evaluate_channel(tr, env, None, n_steps, seed, target)
            setattr(env, flag, False)
    return res, rec0


def tost_equivalence(res, cond_a, cond_b, dv, margin, n_boot=4000, seed=0):
    """Two one-sided tests: is the paired difference inside +/- margin?

    Reporting a non-significant difference is not evidence of no effect.  This
    asks the question that matters instead -- whether the effect is small enough
    to be practically nil -- against a margin fixed in advance as a fraction of
    the intact condition's own value.
    """
    d = res[cond_a]["_per_arena"][dv] - res[cond_b]["_per_arena"][dv]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [5, 95])          # 90% CI == two one-sided 5% tests
    return {"mean": float(d.mean()), "lo90": float(lo), "hi90": float(hi),
            "margin": float(margin),
            "equivalent": bool(lo > -margin and hi < margin)}


def paired_diff(res, cond_a, cond_b, dv, n_boot=2000, seed=0):
    """Within-arena paired difference cond_a - cond_b, with a bootstrap CI.

    Both conditions were run from the same episode seed, so arena k has the same
    initial food layout and the same sensory-noise stream in both; differencing
    within arena removes essentially all of the layout variance.
    """
    a = res[cond_a]["_per_arena"][dv]
    b = res[cond_b]["_per_arena"][dv]
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p_two_sided": float(2 * min((bs <= 0).mean(), (bs >= 0).mean())),
            "n": int(len(d))}


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def decompose_muting(res: dict) -> dict:
    """Split the confounded muting effect into private and public shares.

    mute_both removes both consequences; mute_self removes only the emitter's
    reafference; mute_social removes only the conspecifics' access.  If the two
    effects were additive, their sum equals the joint effect; the residual is
    the interaction.
    """
    out = {}
    for dv in ("eaten_target", "eaten_group", "eaten_others", "nn_dist", "reward_target", "reward_others"):
        both = paired_diff(res, "mute_both", "intact", dv)
        # losing only the emitter's own reafference, public reach intact
        priv = paired_diff(res, "mute_self", "intact", dv)
        # losing only the neighbours' access, reafference intact
        soc = paired_diff(res, "mute_social", "intact", dv)
        # `mute_signal` runs the pulse in cue_only mode: it still illuminates
        # neighbours but they cannot detect it, so the difference isolates the
        # loss of the *signal*.  `mute_cue` is the mirror image.
        sig = paired_diff(res, "mute_signal", "intact", dv) if "mute_signal" in res else None
        cue = paired_diff(res, "mute_cue", "intact", dv) if "mute_cue" in res else None
        out[dv] = {
            "total": both, "private_share": priv, "social_share": soc,
            "cue_share": cue, "signal_share": sig,
            "interaction": both["mean"] - (priv["mean"] + soc["mean"]),
            "social_frac": soc["mean"] / both["mean"] if abs(both["mean"]) > 1e-9 else float("nan"),
        }
    return out


def signal_value(res: dict) -> dict:
    """Payoff consequences of the signal's *contingency*, not its mere presence.

    Comparing the intact channel with a rate-matched replay isolates what the
    receiver gains (or loses) from the pulse being contingent on the emitter's
    actual situation.  Following Maynard Smith & Harper and Dawkins & Krebs, a
    condition counts as cooperative communication only if both parties gain;
    sender-gain with receiver-loss is manipulation.
    """
    out = {}
    for null in ("replay_cross", "scramble_time", "replay_context"):
        if null not in res:
            continue
        s_ = paired_diff(res, "intact", null, "reward_target")
        r_ = paired_diff(res, "intact", null, "reward_others")
        ds, dr = s_["mean"], r_["mean"]
        sig_s = s_["lo"] > 0 or s_["hi"] < 0
        sig_r = r_["lo"] > 0 or r_["hi"] < 0
        if not (sig_s or sig_r):
            out[null] = {"d_sender": s_, "d_receiver": r_, "label": "no_effect"}
            continue
        if ds > 0 and dr > 0:
            label = "communication"
        elif ds > 0 and dr <= 0:
            label = "manipulation"
        elif ds <= 0 and dr > 0:
            label = "exploited_cue"
        else:
            label = "none"
        out[null] = {"d_sender": s_, "d_receiver": r_, "label": label}
    return out
