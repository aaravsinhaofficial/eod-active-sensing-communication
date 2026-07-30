"""Numerical validation of the GPU reimplementation against upstream WEF NumPy code.

Run from the upstream simulator directory so that its flat imports resolve:

    cd wef_upstream/onpolicy/custom/fish
    python .../scripts/validate_physics.py

Reports max relative error for
  1. monopole fields,
  2. dipole fields,
  3. method-of-images wall reflections,
  4. induced conductor moments,
  5. the full mormyromast (active electrolocation) pipeline,
  6. the full knollenorgan (conspecific detection) pipeline,
  7. the full ampullary (passive DC) pipeline.
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
UPSTREAM = os.environ.get(
    "WEF_UPSTREAM",
    os.path.join(HERE, "..", "..", "wef_upstream", "onpolicy", "custom", "fish"),
)
sys.path.insert(0, UPSTREAM)

import electric as up_electric  # noqa: E402
import electric_scene as up_scene  # noqa: E402
import sensing as up_sensing  # noqa: E402
import cfg as up_cfg  # noqa: E402
from dataclasses import replace, replace as replace_dc  # noqa: E402

from eodcomm import constants as C  # noqa: E402
from eodcomm import physics as P  # noqa: E402

torch.set_default_dtype(torch.float64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
rng = np.random.default_rng(0)
results = {}


def relerr(a, b):
    """Max absolute deviation, normalised by the largest magnitude present.

    Reported this way rather than element-wise, because both pipelines contain
    hard-thresholded receptors whose outputs are exactly zero over most of the
    array; an element-wise ratio there is dominated by 0/0.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b)) / scale)


def T(x):
    return torch.as_tensor(np.asarray(x, float), device=DEV, dtype=DT)


# ---------------------------------------------------------------- 1. monopoles
qpos = rng.uniform(1, 59, (40, 2))
mpos = rng.uniform(1, 59, (6, 2))
mq = rng.uniform(-1, 1, 6) * 1.11e-15
ref = up_electric.measure_electric_field_original(qpos, mpos, mq, None, None)
mine = P.field_from_monopoles(T(qpos)[None], T(mpos)[None], T(mq)[None])[0].cpu().numpy()
results["monopole_field"] = relerr(ref, mine)

# ------------------------------------------------------------------ 2. dipoles
dpos = rng.uniform(1, 59, (5, 2))
dmom = rng.uniform(-1, 1, (5, 2)) * 1e-20
ref = up_electric.measure_electric_field_original(qpos, None, None, dpos, dmom)
mine = P.field_from_dipoles(T(qpos)[None], T(dpos)[None], T(dmom)[None])[0].cpu().numpy()
results["dipole_field"] = relerr(ref, mine)

# ------------------------------------------------------- 3. both + reflections
arena = (60.0, 60.0)
rmp, rmq, rdp, rdm = up_electric.reflect_sources(
    monopole_positions_cm=mpos,
    monopole_charges=mq,
    dipole_positions_cm=dpos,
    dipole_moments=dmom,
    arena_size_cm=arena,
    reflection_scale=C.REFLECTION_SCALE,
    flip_on_reflection=C.FLIP_ON_REFLECTION,
)
ref = up_electric.measure_electric_field_original(
    qpos,
    np.concatenate([mpos, rmp]),
    np.concatenate([mq, rmq]),
    np.concatenate([dpos, rdp]),
    np.concatenate([dmom, rdm]),
)
mp2, mq2 = P.with_images_mono(T(mpos)[None], T(mq)[None], arena, True)
dp2, dm2 = P.with_images_dip(T(dpos)[None], T(dmom)[None], arena, True)
mine = P.measure_field(T(qpos)[None], mp2, mq2, dp2, dm2)[0].cpu().numpy()
results["field_with_wall_images"] = relerr(ref, mine)

# ------------------------------------------------------------ 4. induced dipoles
cpos = rng.uniform(1, 59, (12, 2))
ccon = np.full(12, -0.5)
crad = np.full(12, 0.25)
ref = up_electric.induce_dipoles(mpos, mq, None, None, cpos, ccon, crad)
e_at = P.field_from_monopoles(T(cpos)[None], T(mpos)[None], T(mq)[None])
mine = P.induce_dipoles(e_at, T(ccon)[None], T(crad)[None])[0].cpu().numpy()
results["induced_dipole_moments"] = relerr(ref, mine)

ref_clip = up_electric.clip_conductor_moments(ref.copy(), crad, C.MAX_CHARGE_ALLOWED)
mine_clip = P.clip_moments(T(mine)[None], T(crad)[None], C.MAX_CHARGE_ALLOWED)[0].cpu().numpy()
results["clip_conductor_moments"] = relerr(ref_clip, mine_clip)

# ------------------------------------------ 5-7. full sensor pipelines vs upstream
F, NFOOD = 3, 10
fish_pos = rng.uniform(8, 52, (F, 2))
fish_th = rng.uniform(-np.pi, np.pi, F)
fish_eod = np.array([True, False, True])
food_pos = rng.uniform(3, 57, (NFOOD, 2))
food_th = rng.uniform(0, 2 * np.pi, NFOOD)
sizes = np.array([0.2, 0.5, 0.9])


class _StubEnv:
    num_agents = F
    arena_size = arena


sp = up_sensing.SensingParams(
    knollen_metadata_mode="relative",
    collective_sensing_mode=0,           # self-image only (active electrolocation)
    morm_selfimage_mode=1,
    morm_consimage_mode=1,
    noise_frac_morm=0.0,
    noise_frac_amp=0.0,
    noise_frac_knollen=0.0,
    noise_frac_knollen_metadata=0.0,
    ampullary_intrinsic_only=True,
    max_food_sensing_radius=None,
    num_morm_sets=1,
    noise_frac_amp_cons_eod=0.0,  # upstream default is 0.5; silenced for a deterministic comparison
)
geom = dict(up_cfg.AGENT_GEOMETRY_PARAMS)
geom["num_morm_sets"] = 1
model = up_sensing.DynamicBaselineModel(sp, _StubEnv())
model.num_agents = F
model.electric_scene = up_scene.ElectricScene(arena_size_cm=arena, max_charge_allowed=C.MAX_CHARGE_ALLOWED)
model.agent_electrics = model._build_agent_electrics(geom, up_cfg.AGENT_ELECTRIC_PARAMS, F, arena)
model.set_sensor_modes_episode(up_sensing.SensorModesEpisode())

fish_info = {
    "fish_positions_cm": fish_pos,
    "fish_orientations": fish_th,
    "fish_eods": fish_eod,
    "agent_sizes": sizes,
}
food_info = {
    "food_positions_cm": food_pos,
    "food_orientations": food_th,
    "food_radius": C.FOOD_RADIUS_CM,
}
amp_ref, morm_ref, kno_ref, _, meta_ref = model.sense_step(
    fish_info, food_info, list(range(F)), arena, np.random.default_rng(0)
)

# ---- mine ----
from eodcomm.env import EnvConfig, FishEnv  # noqa: E402

cfg = EnvConfig(
    n_fish=F, n_food=NFOOD, batch=1, arena_cm=arena, device=DEV, dtype=DT,
    noise_frac_morm=0.0, noise_frac_amp=0.0, noise_frac_knollen=0.0,
    noise_frac_knollen_meta=0.0, size_mode="none", predation=0.0,
    collective_sensing=0,
)
env = FishEnv(cfg)
env.pos = T(fish_pos)[None]
env.theta = T(fish_th)[None]
env.food = T(food_pos)[None]
env.food_theta = T(food_th)[None]
env.food_alive = torch.ones((1, NFOOD), dtype=torch.bool, device=DEV)
env.size = T(sizes)[None]
emit = torch.as_tensor(fish_eod, device=DEV)[None]

morm_mine = env._sense_mormyromast(emit, emit)[0].cpu().numpy()
kno_mine, meta_mine, _ = env._sense_knollen(emit)
kno_mine = kno_mine[0].cpu().numpy()
amp_mine = env._sense_ampullary()[0].cpu().numpy()

# upstream morm is zero for non-emitting fish only via CD bookkeeping; compare
# the emitting fish, which is where the reafferent signal actually lives.
em = np.where(fish_eod)[0]
results["mormyromast_pipeline"] = relerr(morm_ref[em], morm_mine[em])
results["knollen_pipeline"] = relerr(kno_ref, kno_mine)
results["ampullary_pipeline"] = relerr(amp_ref, amp_mine)

# baseline calibration constants
results["morm_cd_baseline"] = relerr(
    model.agent_electrics.morm_virtual_baselines[0], env.morm_cd.cpu().numpy()
)
results["amp_intrinsic_baseline"] = relerr(
    model.agent_electrics.amp_intrinsic_baseline, env.amp_intrinsic_baseline.cpu().numpy()
)

# ---- mode 1: collective sensing (self + conspecific images), dynamic baseline
sp1 = replace(sp, collective_sensing_mode=1, subtract_cons_baseline=True)
model1 = up_sensing.DynamicBaselineModel(sp1, _StubEnv())
model1.num_agents = F
model1.electric_scene = up_scene.ElectricScene(arena_size_cm=arena, max_charge_allowed=C.MAX_CHARGE_ALLOWED)
model1.agent_electrics = model1._build_agent_electrics(geom, up_cfg.AGENT_ELECTRIC_PARAMS, F, arena)
model1.set_sensor_modes_episode(up_sensing.SensorModesEpisode())
fi = {"fish_positions_cm": fish_pos, "fish_orientations": fish_th,
      "fish_eods": fish_eod, "agent_sizes": sizes}
fo = {"food_positions_cm": food_pos, "food_orientations": food_th, "food_radius": C.FOOD_RADIUS_CM}
_, morm_ref1, _, _, _ = model1.sense_step(fi, fo, list(range(F)), arena, np.random.default_rng(0))

cfg1 = replace_dc(cfg, collective_sensing=1)
env1 = FishEnv(cfg1)
env1.pos = T(fish_pos)[None]; env1.theta = T(fish_th)[None]
env1.food = T(food_pos)[None]; env1.food_theta = T(food_th)[None]
env1.food_alive = torch.ones((1, NFOOD), dtype=torch.bool, device=DEV); env1.size = T(sizes)[None]
morm_mine1 = env1._sense_mormyromast(emit, emit)[0].cpu().numpy()
results["mormyromast_collective_pipeline"] = relerr(morm_ref1, morm_mine1)

print("\n=== GPU reimplementation vs upstream NumPy: max relative error ===")
for k, v in results.items():
    flag = "OK " if v < 1e-6 else ("~  " if v < 1e-3 else "XX ")
    print(f"  {flag} {k:32s} {v:.3e}")

out = os.path.join(HERE, "..", "results", "physics_validation.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {out}")
