"""Measurement machinery for deciding whether a discharge is a signal.

The quantities implemented here follow three literatures:

  * Jaques et al. (2019) causal influence, but computed *interventionally*.
    Because we own the simulator we can re-render a receiver's sensory input
    under do(emit_j = 1) and do(emit_j = 0) with every other cause held on its
    factual path, so no model-of-other-agents approximation is needed.
  * Lowe et al. (2019) positive signalling and positive listening, with their
    recommended artifact controls.
  * Scott-Phillips (2008) cue/signal distinction, operationalised as the Sender
    Shaping Index: does the emission policy depend on whether receivers can
    hear it at all?
"""

from __future__ import annotations

import numpy as np
import torch

from .env import ChannelSpec
from .ppo import kl_hybrid, sample_action


# ---------------------------------------------------------------------------
# Information-theoretic primitives
# ---------------------------------------------------------------------------

def _counts(x: np.ndarray, y: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)


def mi_plugin(x: np.ndarray, y: np.ndarray, nx: int, ny: int) -> float:
    """Mutual information in bits, Miller-Madow bias-corrected."""
    n = x.size
    if n == 0:
        return 0.0
    c = _counts(x, y, nx, ny)
    p = c / n
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())
    # Miller-Madow: (#occupied cells - #occupied rows - #occupied cols + 1) / (2 n ln2)
    bias = ((c > 0).sum() - (c.sum(1) > 0).sum() - (c.sum(0) > 0).sum() + 1) / (2 * n * np.log(2))
    return max(0.0, mi - bias)


def cmi_plugin(x, y, z, nx, ny, nz_) -> float:
    """Conditional MI I(X;Y|Z), averaged over strata Z."""
    tot, n = 0.0, x.size
    for k in range(nz_):
        m = z == k
        if m.sum() < 20:
            continue
        tot += m.sum() / n * mi_plugin(x[m], y[m], nx, ny)
    return tot


def quantize(v: np.ndarray, k: int = 8) -> np.ndarray:
    """Quantile binning; robust to arbitrary marginal shapes."""
    v = np.asarray(v, float)
    finite = np.isfinite(v)
    out = np.zeros(v.shape, dtype=np.int64)
    if finite.sum() == 0:
        return out
    qs = np.quantile(v[finite], np.linspace(0, 1, k + 1)[1:-1])
    out[finite] = np.searchsorted(qs, v[finite], side="right")
    return np.clip(out, 0, k - 1)


def permutation_null(fn, x, y, n_perm=200, rng=None):
    rng = rng or np.random.default_rng(0)
    obs = fn(x, y)
    null = np.array([fn(x, y[rng.permutation(y.size)]) for _ in range(n_perm)])
    p = float((null >= obs).mean())
    return obs, float(null.mean()), float(np.quantile(null, 0.975)), p


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect(tr, env, n_steps: int, channel: ChannelSpec | None = None, seed: int = 0,
            deterministic: bool = False):
    """Roll the trained policy and record everything needed downstream."""
    obs = env.reset(seed)
    tr.reset_state()
    B, F, D = env.B, env.F, env.obs_dim
    rec = {k: [] for k in (
        "emit", "heard", "rew", "ate", "struck", "bit", "bitten", "pos", "theta",
        "size", "nearest_food", "food_near10", "pred_dist", "p_emit", "act",
        "dturn", "dspeed", "obs", "hx", "signal", "active", "arrived",
    )}
    for _ in range(n_steps):
        flat = obs.reshape(1, B * F, D)
        hx_in = tr.hx.clone()
        mu, ls, lg, tr.hx = tr.net.policy(flat, tr.hx)
        if deterministic:
            cont = mu[0]
            disc = (torch.sigmoid(lg[0]) > 0.5).to(mu.dtype)
            act = torch.cat([cont, disc * 2 - 1], -1)
        else:
            act, cont, disc = sample_action(mu[0], ls[0], lg[0])
        p_emit = torch.sigmoid(lg[0, :, 0]).reshape(B, F)

        rec["obs"].append(obs.clone())
        rec["hx"].append(hx_in)
        rec["p_emit"].append(p_emit)
        rec["pos"].append(env.pos.clone())
        rec["theta"].append(env.theta.clone())
        rec["size"].append(env.size.clone())
        food = getattr(env, "food", None)
        if food is None:
            rec["food_near10"].append(torch.zeros((B, F), device=env.dev))
        else:
            d_food = torch.cdist(env.pos, food)
            d_food = torch.where(env.food_alive[:, None, :], d_food,
                                 torch.full_like(d_food, 1e6))
            near_r = 10.0 if getattr(env.cfg, "arena_cm", None) else 0.2
            rec["food_near10"].append((d_food < near_r).sum(-1).to(torch.float32))
        if getattr(env.cfg, "predation", 0) > 0 and getattr(env.cfg, "n_pred", 0) > 0:
            rec["pred_dist"].append(torch.cdist(env.pos, env.pred_pos).min(-1).values)
        else:
            rec["pred_dist"].append(torch.full((B, F), 1e6, device=env.dev))

        A = env.act_dim
        nobs, rew, done, info = env.step(act.reshape(B, F, A), channel)
        rec["act"].append(act.reshape(B, F, A))
        rec["emit"].append(info["emit_self"].to(torch.float32))
        rec["signal"].append(info.get("signal", info["emit_self"]).to(torch.float32))
        rec["heard"].append(info["heard"])
        rec["rew"].append(rew)
        rec["ate"].append(info["ate"])
        rec["struck"].append(info["struck"])
        rec["bit"].append(info["bit"])
        rec["bitten"].append(info["bitten"])
        rec["nearest_food"].append(info["nearest_food"])
        if "active" in info:
            rec["active"].append(info["active"])
            rec["arrived"].append(info["arrived"])
        obs = nobs
        if done:
            obs = env.reset()
            tr.reset_state()
    out = {k: torch.stack(v) for k, v in rec.items() if len(v)}
    # egocentric motion, for "intended movement" decoding
    out["dturn"] = out["act"][..., 1]
    out["dspeed"] = out["act"][..., 0]
    return out


# ---------------------------------------------------------------------------
# Causal influence (interventional)
# ---------------------------------------------------------------------------

@torch.no_grad()
def causal_influence(tr, env, n_steps: int = 256, seed: int = 0):
    """Interventional causal influence of each emitter on each receiver.

    At every step the receiver's observation is re-rendered under
    do(emit_j = 1) and do(emit_j = 0), holding its recurrent state and all other
    inputs on the factual path, and we take the KL between the two resulting
    action distributions.  This is the do-operator analogue of Jaques et al.'s
    counterfactual reasoning, exact rather than estimated.

    Returns (F, F) matrix of mean KL in nats, plus a shift-null baseline in
    which the intervened emitter identity is randomly re-assigned.
    """
    obs = env.reset(seed)
    tr.reset_state()
    B, F, D = env.B, env.F, env.obs_dim
    ci = torch.zeros((F, F), device=env.dev)
    ci_null = torch.zeros((F, F), device=env.dev)
    n = 0
    for _ in range(n_steps):
        flat = obs.reshape(1, B * F, D)
        hx_in = tr.hx.clone()
        mu, ls, lg, tr.hx = tr.net.policy(flat, tr.hx)
        act, cont, disc = sample_action(mu[0], ls[0], lg[0])
        emit = (act[..., 2] > 0).reshape(B, F)

        gstate = env.g.get_state()
        for j in range(F):
            e1 = emit.clone(); e1[:, j] = True
            e0 = emit.clone(); e0[:, j] = False
            env.g.set_state(gstate)
            o1 = env._observe(emit, e1)
            env.g.set_state(gstate)
            o0 = env._observe(emit, e0)
            # Null: re-render the *same* intervention with an independent draw of
            # sensory noise.  This is the divergence the receiver's policy shows
            # when nothing about the message changed, i.e. the noise floor that
            # any apparent influence must clear.
            o1b = env._observe(emit, e1)
            m1, l1, g1, _ = tr.net.policy(o1.reshape(1, B * F, D), hx_in)
            m0, l0, g0, _ = tr.net.policy(o0.reshape(1, B * F, D), hx_in)
            mb, lb, gb, _ = tr.net.policy(o1b.reshape(1, B * F, D), hx_in)
            kl = kl_hybrid(m1[0], l1[0], g1[0], m0[0], l0[0], g0[0]).reshape(B, F)
            kl_n = kl_hybrid(m1[0], l1[0], g1[0], mb[0], lb[0], gb[0]).reshape(B, F)
            ci[j] += kl.mean(0)
            ci_null[j] += kl_n.mean(0)
        env.g.set_state(gstate)
        n += 1
        obs, _, done, _ = env.step(act.reshape(B, F, env.act_dim))
        if done:
            obs = env.reset(); tr.reset_state()
    ci, ci_null = (ci / n).cpu().numpy(), (ci_null / n).cpu().numpy()
    np.fill_diagonal(ci, np.nan)
    np.fill_diagonal(ci_null, np.nan)
    return ci, ci_null


@torch.no_grad()
def positive_listening(tr, env, n_steps: int = 256, seed: int = 0):
    """Lowe et al. positive listening: L1 policy divergence under intervention.

    Uses the time-shift null recommended by Lowe et al.: instead of comparing
    against a silent channel (which is out of distribution), we compare the
    factual message against the same emitter's train taken from a random
    circular time shift, preserving its marginal rate.
    """
    obs = env.reset(seed)
    tr.reset_state()
    B, F, D = env.B, env.F, env.obs_dim
    hist = []
    pl, pl_null, n = 0.0, 0.0, 0
    for t in range(n_steps):
        flat = obs.reshape(1, B * F, D)
        hx_in = tr.hx.clone()
        mu, ls, lg, tr.hx = tr.net.policy(flat, tr.hx)
        act, _, _ = sample_action(mu[0], ls[0], lg[0])
        emit = (act[..., 2] > 0).reshape(B, F)
        hist.append(emit.clone())

        gstate = env.g.get_state()
        env.g.set_state(gstate); o_fact = env._observe(emit, emit)
        shifted = hist[max(0, t - 83)] if len(hist) > 83 else emit[torch.randperm(B, device=env.dev)]
        env.g.set_state(gstate); o_shift = env._observe(emit, shifted)
        env.g.set_state(gstate); o_zero = env._observe(emit, torch.zeros_like(emit))

        pf = tr.net.policy(o_fact.reshape(1, B * F, D), hx_in)
        ps = tr.net.policy(o_shift.reshape(1, B * F, D), hx_in)
        pz = tr.net.policy(o_zero.reshape(1, B * F, D), hx_in)

        def l1(a, b):
            d = (a[0][0] - b[0][0]).abs().sum(-1)
            d = d + (torch.sigmoid(a[2][0]) - torch.sigmoid(b[2][0])).abs().sum(-1)
            return d.mean().item()

        pl += l1(pf, ps)
        pl_null += l1(pf, pz)
        env.g.set_state(gstate)
        n += 1
        obs, _, done, _ = env.step(act.reshape(B, F, env.act_dim))
        if done:
            obs = env.reset(); tr.reset_state(); hist = []
    return pl / n, pl_null / n


# ---------------------------------------------------------------------------
# Positive signalling and message content
# ---------------------------------------------------------------------------

def _window_count(emit: np.ndarray, w: int = 21) -> np.ndarray:
    """Number of discharges in a trailing window -- the pulse-rate code."""
    T = emit.shape[0]
    out = np.zeros_like(emit, dtype=float)
    c = np.cumsum(np.concatenate([np.zeros((1,) + emit.shape[1:]), emit], 0), 0)
    for t in range(T):
        lo = max(0, t - w + 1)
        out[t] = c[t + 1] - c[lo]
    return out


def signal_bit_metrics(rec: dict, w: int = 21, nbin: int = 6):
    """Positive signalling carried by the *decoupled* variable.

    The subtype bit is conditioned on the pulse having been emitted at all, so
    that we measure how the sender modulates content rather than how often it
    probes.  Geometry is stratified out as in `positive_signaling`.
    """
    sig = rec["signal"].cpu().numpy()
    emit = rec["emit"].cpu().numpy()
    if sig.sum() == 0 or np.allclose(sig, emit):
        return None
    pos = rec["pos"].cpu().numpy()
    T, B, F = emit.shape
    # fraction of the window's pulses that carried the subtype
    cnt_s = _window_count(sig, w)
    cnt_e = _window_count(emit, w)
    frac = np.where(cnt_e > 0, cnt_s / np.maximum(cnt_e, 1e-9), 0.0)
    msg = quantize(frac.reshape(-1), nbin)

    d = np.linalg.norm(pos[:, :, :, None, :] - pos[:, :, None, :, :], axis=-1)
    d = d + np.eye(F)[None, None] * 1e6
    strat = quantize(d.min(-1).reshape(-1), 5)

    out = {"rate": float(sig.sum() / max(emit.sum(), 1e-9))}
    rng = np.random.default_rng(0)
    for k, v in (("food", rec["food_near10"].cpu().numpy().reshape(-1)),
                 ("dominance", rec["size"].cpu().numpy().reshape(-1)),
                 ("movement", rec["dturn"].cpu().numpy().reshape(-1))):
        if np.nanstd(v) == 0:
            out[k] = {"mi": 0.0, "null": 0.0, "p": 1.0}
            continue
        yq = quantize(v, nbin)
        f = lambda a, b: cmi_plugin(a, b, strat, nbin, nbin, 5)  # noqa: E731
        mi = f(msg, yq)
        null = np.array([f(msg, yq[rng.permutation(yq.size)]) for _ in range(40)])
        out[k] = {"mi": float(mi), "null": float(null.mean()),
                  "p": float((null >= mi).mean())}
    return out


def positive_signaling(rec: dict, w: int = 21, nbin: int = 6, n_perm: int = 200):
    """I(message ; sender's private state) / H(state), stratified by geometry.

    Stratification by distance to the nearest conspecific is essential: a fish
    that is near a neighbour is also near food in a patchy world, so an
    unconditioned MI can be manufactured entirely by geometry.
    """
    emit = rec["emit"].cpu().numpy()                        # (T,B,F)
    pos = rec["pos"].cpu().numpy()
    T, B, F = emit.shape
    msg = quantize(_window_count(emit, w).reshape(-1), nbin)

    d = np.linalg.norm(pos[:, :, :, None, :] - pos[:, :, None, :, :], axis=-1)
    d = d + np.eye(F)[None, None] * 1e6
    strat = quantize(d.min(-1).reshape(-1), 5)

    targets = {
        "food": rec["food_near10"].cpu().numpy().reshape(-1),
        "dominance": rec["size"].cpu().numpy().reshape(-1),
        "movement": rec["dturn"].cpu().numpy().reshape(-1),
        "danger": -rec["pred_dist"].cpu().numpy().reshape(-1),
    }
    out = {}
    rng = np.random.default_rng(0)
    for k, v in targets.items():
        if not np.isfinite(v).any() or np.nanstd(v) == 0:
            out[k] = {"mi": 0.0, "null": 0.0, "p": 1.0, "norm": 0.0}
            continue
        yq = quantize(v, nbin)
        f = lambda a, b: cmi_plugin(a, b, strat, nbin, nbin, 5)  # noqa: E731
        mi = f(msg, yq)
        null = np.array([f(msg, yq[rng.permutation(yq.size)]) for _ in range(max(20, n_perm // 4))])
        hy = mi_plugin(yq, yq, nbin, nbin)
        out[k] = {
            "mi": float(mi),
            "null": float(null.mean()),
            "p": float((null >= mi).mean()),
            "norm": float(max(0.0, mi - null.mean()) / max(hy, 1e-9)),
        }
    return out


def decode_content(rec: dict, w: int = 21, horizon: int = 41):
    """How much can a *receiver* recover about the emitter's world?

    The predictors are strictly receiver-observable: the pulse count the
    receiver heard from that emitter over a trailing window, and whether it
    heard anything at all.  Ground-truth fields are never given to the decoder.
    Reported as cross-validated R^2 (continuous) or AUC (binary) minus the same
    quantity computed on time-shifted messages.
    """
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import roc_auc_score

    heard = rec["heard"].cpu().numpy()          # (T,B,F,F-1) receiver x emitter slot
    emit = rec["emit"].cpu().numpy()            # (T,B,F)
    T, B, F = emit.shape
    hc = _window_count(heard.reshape(T, -1), w).reshape(T, B, F, F - 1)

    # map slot -> emitter index
    slots = np.array([[j for j in range(F) if j != i] for i in range(F)])
    res = {}
    rng = np.random.default_rng(0)

    def build(tgt_full, shift=0):
        X, Y = [], []
        for i in range(F):
            for s, j in enumerate(slots[i]):
                x = hc[:, :, i, s]
                if shift:
                    x = np.roll(x, shift, axis=0)
                y = tgt_full[:, :, j]
                X.append(np.stack([x.reshape(-1), (x > 0).astype(float).reshape(-1)], 1))
                Y.append(y.reshape(-1))
        return np.concatenate(X), np.concatenate(Y)

    fut = np.zeros_like(emit)
    fut[:-horizon] = np.array([
        rec["ate"].cpu().numpy()[t:t + horizon].sum(0) for t in range(T - horizon)
    ])
    targets = {
        "food": (rec["food_near10"].cpu().numpy(), "r2"),
        "will_eat": ((fut > 0).astype(float), "auc"),
        "dominance": (rec["size"].cpu().numpy(), "r2"),
        "movement": (rec["dturn"].cpu().numpy(), "r2"),
        "danger": ((rec["pred_dist"].cpu().numpy() < 25).astype(float), "auc"),
    }
    for name, (tgt, kind) in targets.items():
        if np.std(tgt) < 1e-9 or (kind == "auc" and len(np.unique(tgt)) < 2):
            res[name] = {"score": 0.0, "null": 0.0, "delta": 0.0}
            continue
        scores = []
        for sh in (0, 200):
            X, Y = build(tgt, shift=sh)
            n = len(Y)
            idx = rng.permutation(n)
            tr_i, te_i = idx[: n // 2], idx[n // 2:]
            if kind == "r2":
                m = Ridge(alpha=1.0).fit(X[tr_i], Y[tr_i])
                p = m.predict(X[te_i])
                s = 1 - np.mean((Y[te_i] - p) ** 2) / max(np.var(Y[te_i]), 1e-12)
            else:
                if len(np.unique(Y[tr_i])) < 2:
                    s = 0.5
                else:
                    m = LogisticRegression(max_iter=200).fit(X[tr_i], Y[tr_i])
                    s = roc_auc_score(Y[te_i], m.predict_proba(X[te_i])[:, 1])
            scores.append(float(s))
        res[name] = {"score": scores[0], "null": scores[1], "delta": scores[0] - scores[1], "kind": kind}
    return res


# ---------------------------------------------------------------------------
# Sender Shaping Index: the cue / signal discriminator
# ---------------------------------------------------------------------------

def knollen_slice(env):
    """Index range of the knollen block (plus its size metadata) in the observation."""
    from . import constants as C
    a = C.NUM_MORM + C.NUM_AMP
    b = a + C.NUM_KNOLLEN * (env.F - 1) + (env.F - 1)
    return a, b


@torch.no_grad()
def sender_shaping_index(nets_hearing, nets_deaf, ctx_obs, device="cuda", mask=None):
    """SSI = between-condition divergence / within-condition seed divergence - 1.

    Two populations are trained identically except that in one, receivers can
    hear (knollen intact) and in the other they cannot.  If the emission policy
    is merely a probe schedule -- a cue -- it should not care whether anyone is
    listening, and SSI collapses to zero.  If emission has been shaped by its
    effect on receivers, SSI is positive.  This is Scott-Phillips's "produced
    because of that effect" clause, made measurable.

    `ctx_obs` is a common, condition-neutral batch of observations so both
    populations are evaluated on identical inputs.  `mask` should zero the
    knollen block: a deaf-world policy never saw a non-zero knollen input during
    training, so evaluating it on one would measure out-of-distribution
    extrapolation rather than a difference in emission policy.  With the block
    zeroed, both populations are inside the support they were trained on (a
    hearing agent sees an all-zero knollen block whenever no conspecific has just
    discharged), and any remaining divergence is a genuine difference in how the
    two worlds decided to schedule discharges.
    """
    if mask is not None:
        ctx_obs = ctx_obs.clone()
        ctx_obs[:, mask[0]:mask[1]] = 0.0
    def emit_p(net, obs):
        h = torch.zeros((1, obs.shape[0], net.gru.hidden_size), device=device)
        _, _, lg, _ = net.policy(obs[None], h)
        return torch.sigmoid(lg[0, :, 0])

    def jsd(p, q):
        m = 0.5 * (p + q)
        def kl(a, b):
            return (a * (torch.log(a + 1e-8) - torch.log(b + 1e-8))
                    + (1 - a) * (torch.log(1 - a + 1e-8) - torch.log(1 - b + 1e-8)))
        return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean().item()

    ph = [emit_p(n, ctx_obs) for n in nets_hearing]
    pd = [emit_p(n, ctx_obs) for n in nets_deaf]
    between = np.mean([[jsd(a, b) for b in pd] for a in ph])
    within = []
    for grp in (ph, pd):
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                within.append(jsd(grp[i], grp[j]))
    within = float(np.mean(within)) if within else float("nan")
    return {
        "ssi": float(between / max(within, 1e-9) - 1.0),
        "between": float(between),
        "within": within,
    }
