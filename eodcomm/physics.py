"""Batched electrostatics on the GPU.

Every function is vectorised over a leading batch dimension `B` so that
thousands of independent arenas can be simulated in lockstep.  The equations
follow the upstream NumPy reference (`onpolicy/custom/fish/electric.py`)
term-for-term, including its distance softening convention

    d = |r - r'| + eps_m

which is applied *before* the cube/fifth power, and the first-order method of
images used for the non-conducting arena walls.
"""

from __future__ import annotations

import torch

from . import constants as C


def field_from_monopoles(
    query_cm: torch.Tensor,  # (B, Q, 2)
    src_cm: torch.Tensor,  # (B, P, 2)
    charge: torch.Tensor,  # (B, P)
    eps_m: float = C.EPS_M,
) -> torch.Tensor:
    """Coulomb field of a set of point charges.

    E(r) = k * sum_p q_p (r - r_p) / (|r - r_p| + eps)^3
    """
    if src_cm.shape[1] == 0:
        return torch.zeros_like(query_cm)
    off = (query_cm[:, :, None, :] - src_cm[:, None, :, :]) * C.CM_TO_M  # (B,Q,P,2)
    dist = off.norm(dim=-1) + eps_m  # (B,Q,P)
    inv3 = dist.pow(-3)
    return C.K_COULOMB * torch.einsum("bqp,bqp,bqpd->bqd", charge[:, None, :].expand(-1, off.shape[1], -1), inv3, off)


def field_from_dipoles(
    query_cm: torch.Tensor,  # (B, Q, 2)
    src_cm: torch.Tensor,  # (B, D, 2)
    moment: torch.Tensor,  # (B, D, 2)
    eps_m: float = C.EPS_M,
) -> torch.Tensor:
    """Field of ideal point dipoles.

    E(r) = k * sum_d [ 3 (p_d . u) u / |u|^5 - p_d / |u|^3 ],  u = r - r_d
    """
    if src_cm.shape[1] == 0:
        return torch.zeros_like(query_cm)
    off = (query_cm[:, :, None, :] - src_cm[:, None, :, :]) * C.CM_TO_M  # (B,Q,D,2)
    dist = off.norm(dim=-1) + eps_m  # (B,Q,D)
    inv3 = dist.pow(-3)
    inv5 = dist.pow(-5)
    p = moment[:, None, :, :]  # (B,1,D,2)
    p_dot_u = (p * off).sum(-1)  # (B,Q,D)
    term = 3.0 * (p_dot_u * inv5)[..., None] * off - inv3[..., None] * p
    return C.K_COULOMB * term.sum(dim=2)


def measure_field(
    query_cm: torch.Tensor,
    mono_cm: torch.Tensor | None,
    mono_q: torch.Tensor | None,
    dip_cm: torch.Tensor | None,
    dip_p: torch.Tensor | None,
    eps_m: float = C.EPS_M,
) -> torch.Tensor:
    out = torch.zeros_like(query_cm)
    if mono_cm is not None:
        out = out + field_from_monopoles(query_cm, mono_cm, mono_q, eps_m)
    if dip_cm is not None:
        out = out + field_from_dipoles(query_cm, dip_cm, dip_p, eps_m)
    return out


# ---------------------------------------------------------------------------
# Method of images (first order, four walls of a rectangular arena)
# ---------------------------------------------------------------------------

def reflect_monopoles(pos_cm: torch.Tensor, q: torch.Tensor, arena_cm):
    """Return the four first-order image sets, concatenated along the source axis."""
    w, h = arena_cm
    p = pos_cm
    imgs = [
        torch.stack([-p[..., 0], p[..., 1]], dim=-1),  # left wall  (x = 0)
        torch.stack([2 * w - p[..., 0], p[..., 1]], dim=-1),  # right wall
        torch.stack([p[..., 0], -p[..., 1]], dim=-1),  # bottom wall (y = 0)
        torch.stack([p[..., 0], 2 * h - p[..., 1]], dim=-1),  # top wall
    ]
    scale = C.REFLECTION_SCALE * (-1.0 if C.FLIP_ON_REFLECTION else 1.0)
    return torch.cat(imgs, dim=1), torch.cat([q * scale] * 4, dim=1)


def reflect_dipoles(pos_cm: torch.Tensor, p_mom: torch.Tensor, arena_cm):
    w, h = arena_cm
    p = pos_cm
    pos_imgs = [
        torch.stack([-p[..., 0], p[..., 1]], dim=-1),
        torch.stack([2 * w - p[..., 0], p[..., 1]], dim=-1),
        torch.stack([p[..., 0], -p[..., 1]], dim=-1),
        torch.stack([p[..., 0], 2 * h - p[..., 1]], dim=-1),
    ]
    if C.FLIP_ON_REFLECTION:
        flip_x = torch.stack([-p_mom[..., 0], p_mom[..., 1]], dim=-1)
        flip_y = torch.stack([p_mom[..., 0], -p_mom[..., 1]], dim=-1)
        mom_imgs = [flip_x, flip_x, flip_y, flip_y]
    else:
        mom_imgs = [p_mom] * 4
    scale = C.REFLECTION_SCALE
    return torch.cat(pos_imgs, dim=1), torch.cat(mom_imgs, dim=1) * scale


def with_images_mono(pos_cm, q, arena_cm, use_reflections: bool):
    if not use_reflections:
        return pos_cm, q
    rp, rq = reflect_monopoles(pos_cm, q, arena_cm)
    return torch.cat([pos_cm, rp], dim=1), torch.cat([q, rq], dim=1)


def with_images_dip(pos_cm, p_mom, arena_cm, use_reflections: bool):
    if not use_reflections:
        return pos_cm, p_mom
    rp, rm = reflect_dipoles(pos_cm, p_mom, arena_cm)
    return torch.cat([pos_cm, rp], dim=1), torch.cat([p_mom, rm], dim=1)


# ---------------------------------------------------------------------------
# Induced dipoles on conductors (food, non-emitting fish bodies)
# ---------------------------------------------------------------------------

def induce_dipoles(
    e_at_conductor: torch.Tensor,  # (B, N, 2) V/m
    contrast: torch.Tensor,  # (B, N)
    radius_cm: torch.Tensor,  # (B, N)
) -> torch.Tensor:
    """Polarisation of a small sphere in an applied field.

    m = 3 eps_0 V chi E,  V = 4/3 pi a^3   (a in metres)
    """
    a = radius_cm * C.CM_TO_M
    volume = (4.0 / 3.0) * torch.pi * a.pow(3)
    return 3.0 * C.EPSILON_0 * (volume * contrast)[..., None] * e_at_conductor


def clip_moments(moment: torch.Tensor, radius_cm: torch.Tensor, max_charge: float):
    """Upstream `clip_conductor_moments`: cap |m| at q_max * radius (radius in cm)."""
    mag = moment.norm(dim=-1)
    max_mag = max_charge * radius_cm
    factor = torch.where(mag > 0, (max_mag / mag.clamp_min(1e-300)).clamp(max=1.0), torch.ones_like(mag))
    return moment * factor[..., None]


# ---------------------------------------------------------------------------
# Frame transforms and sensor projection
# ---------------------------------------------------------------------------

def rotate(vec_ego: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate (..., S, 2) ego vectors by per-agent angles theta (...,)."""
    c = torch.cos(theta)[..., None]  # (..., 1) broadcasts against the sensor axis
    s = torch.sin(theta)[..., None]
    x, y = vec_ego[..., 0], vec_ego[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


def rotate_one(vec_ego: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate a single (..., 2) vector per element by theta (...)."""
    return rotate(vec_ego.unsqueeze(-2), theta).squeeze(-2)


def to_world(vec_ego: torch.Tensor, theta: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    return rotate(vec_ego, theta) + origin[..., None, :]


def project(field: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    return (field * normal).sum(-1)


# ---------------------------------------------------------------------------
# Receptor transduction
# ---------------------------------------------------------------------------

def process_sensor_readings(
    x: torch.Tensor, smin: float, smax: float, min_magnitude: float = C.GENERAL_SENSOR_MIN
) -> torch.Tensor:
    """Sign-preserving logarithmic compression into [-1, 1] (upstream identical)."""
    sign = torch.sign(x)
    mag = x.abs().clamp(min=smin, max=smax)
    log_min, log_max = torch.log10(torch.tensor(smin)), torch.log10(torch.tensor(smax))
    lm = torch.log10(mag.clamp_min(min_magnitude))
    return sign * (lm - log_min.to(x)) / (log_max - log_min).to(x)


def process_morm(raw_minus_baseline: torch.Tensor) -> torch.Tensor:
    """Mormyromast transduction (upstream DynamicBaselineModel: fixed sensor range)."""
    return process_sensor_readings(raw_minus_baseline, C.MORM_SENSOR_MIN, C.MORM_SENSOR_MAX)


def binarize(x: torch.Tensor, threshold: float, sensor_min: float) -> torch.Tensor:
    """Knollenorgan transduction: an all-or-none spike whose sign encodes field polarity."""
    sign = torch.sign(x)
    sign = torch.where(x.abs() > sensor_min, sign, torch.zeros_like(sign))
    return sign * (x.abs() > threshold).to(x.dtype)
