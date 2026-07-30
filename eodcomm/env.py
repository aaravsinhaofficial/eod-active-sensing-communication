"""Batched multi-agent electric-fish foraging environment.

A GPU reimplementation of the WEF simulator of Singh, Johnson-Yu et al. (2025),
extended with

  * counterfactual EOD channels that dissociate the private (reafferent) and
    public (conspecific-detectable) consequences of a discharge,
  * a metabolic cost per discharge,
  * electroreceptive predators that home in on EODs,
  * switchable cooperative / competitive food economics,
  * configurable knollenorgan (social receptor) sensitivity, i.e. signal range.

Every quantity carries a leading batch axis B (independent arenas) so that the
whole population is stepped as a single set of tensor ops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import constants as C
from . import physics as P


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    n_fish: int = 4
    n_food: int = 48
    n_patches: int = 4
    patch_sigma_cm: float = 6.0
    arena_cm: tuple = (60.0, 60.0)
    episode_len: int = 400
    batch: int = 512
    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    # --- economics -------------------------------------------------------
    eod_cost: float = 0.0          # reward units subtracted per discharge
    shared_food: float = 0.0       # 0 = pure competition, 1 = fully shared harvest
    food_exclusive: bool = True    # an eaten item is removed for everyone

    # --- predation -------------------------------------------------------
    predation: float = 0.0         # penalty magnitude when struck
    n_pred: int = 1
    pred_detect_cm: float = 40.0   # radius over which a discharge is detectable
    pred_speed_cm: float = 0.30    # cm per step
    pred_strike_cm: float = 4.0
    pred_cooldown: int = 40

    # --- social channel --------------------------------------------------
    knollen_gain: float = 1.0      # multiplies the knollen field -> sets social range
    knollen_enabled: bool = True
    eod_allowed: bool = True       # False = permanently mute (passive-sensing control)
    # 0 = self-image only, 1 = self + conspecific images (upstream default),
    # 2 = conspecific images only
    collective_sensing: int = 1
    illuminate_others: bool = True  # do my pulses light up my neighbours' world?
    # Training-time control for the Sender Shaping Index: receivers still get
    # knollen input with the same statistics, but sender identity is permuted
    # every step, so no sender's pulses contingently reach any particular
    # receiver.  A sender that only cares about *having* social input should
    # behave as in the hearing world; one that cares about being listened to
    # should behave as in the deaf world.
    scramble_id_always: bool = False

    # --- the decoupled signalling variable (positive control) -------------
    # A second Bernoulli action: a discharge "subtype" that rides on the pulse.
    # It reaches conspecific knollenorgans but changes nothing about the
    # emitter's own reafference, nothing about how the pulse illuminates
    # neighbours, and nothing about predator detectability.  Sending it still
    # requires paying for a pulse, so the coupling is partial, not absent: what
    # is decoupled is the *content*, not the act.
    signal_channel: bool = False
    signal_cost: float = 0.0        # bandwidth price, on top of the pulse cost

    # --- yoked reception (the sender-shaping control) ---------------------
    # Receivers get knollen input drawn from another arena's emission mask, so
    # reception statistics are matched to the live world while no agent's own
    # pulses reach anybody.  Contrast with the live world isolates being
    # listened to from merely having social input -- which `knollen_enabled`
    # does not, because it removes reception rather than audibility.
    yoked_knollen: bool = False

    # --- referential assay -------------------------------------------------
    # A Lewis signalling game embedded in the same physics.  One immobile fish
    # privately observes which of two sites holds food; the other must go to a
    # site.  The sender's discharges are forced onto a fixed schedule, so pulse
    # timing carries nothing and the subtype bit is the only free channel.
    # Proximity shaping is disabled because it is computed from ground-truth
    # food distance and would leak the answer to the receiver.
    task: str = "forage"            # 'forage' | 'referential'
    ref_sites: tuple = ((15.0, 55.0), (75.0, 55.0))
    ref_sender_xy: tuple = (45.0, 8.0)
    ref_start_xy: tuple = (45.0, 16.0)
    ref_sender_emits: bool = True   # force the sender's pulse schedule
    ref_trial_len: int = 128        # many trials per episode
    ref_at_site_cm: float = 12.0    # scoring radius; well inside the 60 cm gap
    ref_site_shaping: float = 3.0   # shaping toward the nearest *site*: symmetric
                                    # between the two, so it leaks no answer
    # Ground-truth positive control.  With a scripted sender the existence of
    # communication is not in question -- the subtype simply *is* the private
    # cue -- so the experiment tests whether our assays detect communication
    # known to be present, rather than whether policy gradient can invent it.
    #   'none'    the sender learns what to send (the emergent condition)
    #   'honest'  subtype := active site
    #   'random'  subtype := a fair coin, rate-matched, carrying nothing
    ref_scripted: str = "none"
    # Score the referential task on reaching the site the signal names, once per
    # trial.  Scoring it on food eaten would confound signal use with
    # close-range foraging skill, which is not what is under test.
    ref_arrival_reward: float = 10.0
    # Mediation intervention.  The transmitted subtype is computed from an
    # independently resampled referent while the true referent -- and therefore
    # the food, and every non-message pathway from sender state to receiver
    # payoff -- is left exactly as it was.  This realises Pearl's natural
    # indirect effect: it asks what the receiver loses when the message stops
    # tracking the world, without ever taking the message out of its own
    # marginal distribution, which is what deleting it would do.
    ref_mediate: bool = False

    morm_enabled: bool = True
    amp_enabled: bool = True

    # --- morphology / identity ------------------------------------------
    size_mode: str = "hierarchy"   # 'hierarchy' | 'uniform' | 'none'
    persistent_identity: bool = True

    # --- noise -----------------------------------------------------------
    noise_frac_morm: float = 0.05
    noise_frac_amp: float = 0.05
    noise_frac_amp_cons_eod: float = 0.50  # a conspecific discharge jams passive sensing
    noise_frac_knollen: float = 0.05
    noise_frac_knollen_meta: float = 0.05

    # --- reward ----------------------------------------------------------
    r_eat: float = C.REWARDS["eat"]
    r_shaping: float = C.REWARDS["proximity_shaping"]
    r_bitten: float = C.REWARDS["bitten"]
    r_bite: float = C.REWARDS["bite"]
    r_collision: float = C.REWARDS["collision"]

    use_reflections: bool = True


@dataclass
class ChannelSpec:
    """A counterfactual manipulation of the dual-use discharge.

    `mode` selects which of the two consequences of a pulse survives:

      intact        both  — the discharge illuminates the emitter's world and is
                            heard by conspecifics (the biological default)
      private       self  — reafference only; the pulse neither illuminates nor
                            is heard by conspecifics
      cue_only      the pulse still illuminates conspecifics' electrolocation
                    (they exploit the probe) but is invisible to their
                    knollenorgans, so it cannot act as a detected signal
      signal_only   conspecifics detect the pulse but gain no illumination from
                    it, isolating the symbolic channel
      social        other — conspecifics hear it; the emitter gains no reafference
      phantom       public input is generated by an exogenous Bernoulli process
                    that no agent chose
      replay        public input is a recorded pulse train from a different
                    behavioural context (marginal rate preserved, contingency
                    with the emitter's current state destroyed)
      scramble_time public input is a random temporal permutation of the true
                    train (rate preserved, precise timing destroyed)
      scramble_id   public input is delivered under permuted sender identities
                    (rate and timing preserved, identity destroyed)
    """

    mode: str = "intact"
    agents: tuple | None = None     # which emitters are manipulated; None = all
    phantom_rate: float = 0.15
    replay_train: torch.Tensor | None = None   # (B, T, F) bool
    seed: int = 0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FishEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.dt_type = cfg.dtype
        B, F = cfg.batch, cfg.n_fish
        self.B, self.F = B, F

        self._build_geometry()
        self._build_baselines()

        self.n_knollen_slots = F - 1
        self.obs_dim = (
            C.NUM_MORM
            + C.NUM_AMP
            + C.NUM_KNOLLEN * self.n_knollen_slots
            + self.n_knollen_slots      # emitter size metadata
            + 4                          # last action (tanh)
            + 1                          # was_bitten
            + 1                          # own size
            + 1                          # bite cooldown
            + 1                          # eat cooldown
            + (self.n_knollen_slots if cfg.signal_channel else 0)  # heard subtype
            + (5 if cfg.task == "referential" else 0)   # private cue + 2 site bearings
        )
        self.n_disc = 3 if cfg.signal_channel else 2
        self.act_dim = 2 + self.n_disc  # thrust, turn, emit, bite [, signal]

        self.g = torch.Generator(device=self.dev)
        self.g.manual_seed(0)

    # ------------------------------------------------------------------
    # Static geometry and calibration
    # ------------------------------------------------------------------
    def _t(self, x):
        return torch.as_tensor(x, device=self.dev, dtype=self.dt_type)

    def _build_geometry(self):
        r = C.BODY_RADIUS_CM
        ma = C.mormyromast_angles()
        ka = C.uniform_angles(C.NUM_KNOLLEN)
        aa = C.uniform_angles(C.NUM_AMP)

        def ring(angles):
            import numpy as np
            pos = np.stack([r * np.cos(angles), r * np.sin(angles)], -1)
            nrm = np.stack([np.cos(angles), np.sin(angles)], -1)
            return self._t(pos), self._t(nrm)

        self.morm_pos_ego, self.morm_nrm_ego = ring(ma)
        self.kno_pos_ego, self.kno_nrm_ego = ring(ka)
        self.amp_pos_ego, self.amp_nrm_ego = ring(aa)
        self.mono_ego = self._t(C.MONOPOLE_POSITIONS_EGO)          # (2,2)
        self.mono_q = self._t(C.MONOPOLE_CHARGES)                  # (2,)
        self.fish_intrinsic_ego = self._t(C.FISH_INTRINSIC_DIPOLE_MOMENT)
        self.food_intrinsic_ego = self._t(C.FOOD_INTRINSIC_DIPOLE_MOMENT)
        self.pred_intrinsic_ego = self._t(C.PRED_INTRINSIC_DIPOLE_MOMENT)

    def _build_baselines(self):
        """Corollary-discharge and intrinsic baselines for a fish at arena centre.

        Upstream computes these once, for a canonical pose, and subtracts them at
        every step; the reafferent signal is therefore the *deviation* of the
        received field from what the fish's own discharge would produce in an
        empty arena.
        """
        cfg = self.cfg
        w, h = cfg.arena_cm
        centre = self._t([[w / 2, h / 2]])                       # (1,2)
        theta = self._t([0.0])

        mono = P.to_world(self.mono_ego[None], theta, centre)      # (1,2,2)
        q = self.mono_q[None]                                      # (1,2)
        mp, mq = P.with_images_mono(mono, q, cfg.arena_cm, cfg.use_reflections)

        # --- mormyromast CD ---
        mq_pos = P.to_world(self.morm_pos_ego[None], theta, centre)
        mq_nrm = P.rotate(self.morm_nrm_ego[None], theta)
        e = P.measure_field(mq_pos, mp, mq, None, None)
        self.morm_cd = P.project(e, mq_nrm)[0]                     # (36,)

        e_ctr = P.measure_field(centre[None], mp, mq, None, None)
        self.center_morm_cd = e_ctr[0, 0]                          # (2,)

        # --- ampullary intrinsic baseline (own DC field) ---
        dip_pos = centre[None]                                     # (1,1,2)
        dip_mom = P.rotate_one(self.fish_intrinsic_ego[None, None], theta)
        dp, dm = P.with_images_dip(dip_pos, dip_mom, cfg.arena_cm, cfg.use_reflections)
        aq_pos = P.to_world(self.amp_pos_ego[None], theta, centre)
        aq_nrm = P.rotate(self.amp_nrm_ego[None], theta)
        e_a = P.measure_field(aq_pos, None, None, dp, dm)
        self.amp_intrinsic_baseline = P.project(e_a, aq_nrm)[0]    # (24,)

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        cfg = self.cfg
        B, F = self.B, self.F
        if seed is not None:
            self.g.manual_seed(seed)
        w, h = cfg.arena_cm
        margin = 5.0

        self.pos = torch.rand((B, F, 2), generator=self.g, device=self.dev, dtype=self.dt_type)
        self.pos[..., 0] = self.pos[..., 0] * (w - 2 * margin) + margin
        self.pos[..., 1] = self.pos[..., 1] * (h - 2 * margin) + margin
        self.theta = (torch.rand((B, F), generator=self.g, device=self.dev, dtype=self.dt_type) * 2 - 1) * math.pi

        # Body size: a stable within-episode dominance rank.  With
        # persistent_identity the rank is tied to the agent index, so slot j of
        # the knollen array always refers to the same individual.
        if cfg.size_mode == "hierarchy":
            base = torch.linspace(0.15, 0.95, F, device=self.dev, dtype=self.dt_type)
            self.size = base[None].expand(B, F).contiguous()
            if not cfg.persistent_identity:
                perm = torch.argsort(torch.rand((B, F), generator=self.g, device=self.dev), dim=1)
                self.size = torch.gather(self.size, 1, perm)
        elif cfg.size_mode == "uniform":
            self.size = torch.rand((B, F), generator=self.g, device=self.dev, dtype=self.dt_type)
        else:
            self.size = torch.full((B, F), 0.5, device=self.dev, dtype=self.dt_type)

        self._spawn_food()
        if cfg.task == "referential":
            self.pos[:, 0] = self._t(cfg.ref_sender_xy)
            self.pos[:, 1:] = self._t(cfg.ref_start_xy)
            self.pos[:, 1:] += (torch.rand(self.pos[:, 1:].shape, generator=self.g,
                                           device=self.dev, dtype=self.dt_type) - 0.5) * 4.0
            self.theta[:] = math.pi / 2

        self.pred_pos = torch.rand((B, cfg.n_pred, 2), generator=self.g, device=self.dev, dtype=self.dt_type)
        self.pred_pos[..., 0] *= w
        self.pred_pos[..., 1] *= h
        self.pred_cd = torch.zeros((B, cfg.n_pred), device=self.dev, dtype=self.dt_type)

        self.last_act = torch.zeros((B, F, 4), device=self.dev, dtype=self.dt_type)
        self._sig = torch.zeros((B, F), device=self.dev, dtype=torch.bool)
        self._arrived = torch.zeros((B, F), device=self.dev, dtype=torch.bool)
        self._rand_bit = (torch.rand((B,), generator=self.g, device=self.dev) < 0.5)
        # an independent copy of the referent, used only to generate the message
        # under the mediation intervention
        self._active_msg = (torch.rand((B,), generator=self.g, device=self.dev) < 0.5).long()
        self.was_bitten = torch.zeros((B, F), device=self.dev, dtype=self.dt_type)
        self.bite_cd = torch.zeros((B, F), device=self.dev, dtype=self.dt_type)
        self.eat_cd = torch.zeros((B, F), device=self.dev, dtype=self.dt_type)
        self.prev_food_dist = torch.full((B, F), float("nan"), device=self.dev, dtype=self.dt_type)
        self.t = 0

        emit0 = torch.zeros((self.B, self.F), device=self.dev, dtype=torch.bool)
        return self._observe(emit0, emit0, None, emit0)

    def _spawn_food(self):
        cfg = self.cfg
        if cfg.task == "referential":
            return self._spawn_referential()
        B, N = self.B, cfg.n_food
        w, h = cfg.arena_cm
        k = cfg.n_patches
        centres = torch.rand((B, k, 2), generator=self.g, device=self.dev, dtype=self.dt_type)
        centres[..., 0] = centres[..., 0] * (w - 20) + 10
        centres[..., 1] = centres[..., 1] * (h - 20) + 10
        which = torch.randint(0, k, (B, N), generator=self.g, device=self.dev)
        base = torch.gather(centres, 1, which[..., None].expand(-1, -1, 2))
        jitter = torch.randn((B, N, 2), generator=self.g, device=self.dev, dtype=self.dt_type) * cfg.patch_sigma_cm
        self.food = (base + jitter).clamp(min=1.0)
        self.food[..., 0] = self.food[..., 0].clamp(max=w - 1)
        self.food[..., 1] = self.food[..., 1].clamp(max=h - 1)
        self.food_alive = torch.ones((B, N), device=self.dev, dtype=torch.bool)
        self.food_theta = torch.rand((B, N), generator=self.g, device=self.dev, dtype=self.dt_type) * 2 * math.pi

    def _spawn_referential(self, which=None):
        cfg = self.cfg
        B, N = self.B, cfg.n_food
        sites = self._t(cfg.ref_sites)                       # (2,2)
        new = (torch.rand((B,), generator=self.g, device=self.dev) < 0.5).long()
        self.active = new if which is None else torch.where(which, new, self.active)
        centre = sites[self.active]                          # (B,2)
        jitter = torch.randn((B, N, 2), generator=self.g, device=self.dev,
                             dtype=self.dt_type) * 2.5
        food = (centre[:, None, :] + jitter)
        w, h = cfg.arena_cm
        food[..., 0] = food[..., 0].clamp(1.0, w - 1)
        food[..., 1] = food[..., 1].clamp(1.0, h - 1)
        alive = torch.ones((B, N), device=self.dev, dtype=torch.bool)
        th = torch.rand((B, N), generator=self.g, device=self.dev, dtype=self.dt_type) * 2 * math.pi
        if which is None:
            self.food, self.food_alive, self.food_theta = food, alive, th
        else:
            m = which[:, None]
            self.food = torch.where(m[..., None], food, self.food)
            self.food_alive = torch.where(m, alive, self.food_alive)
            self.food_theta = torch.where(m, th, self.food_theta)

    def _referential_trial_reset(self):
        """Begin a fresh trial: new active site, fresh food, receiver back at the gate.

        Many short trials per episode rather than one long one.  Each trial is an
        independent signalling event, and that is what makes the game learnable
        in a practical number of steps -- with one trial per episode the sender
        gets a single bit of feedback per 384 steps.
        """
        cfg = self.cfg
        due = (self.t % cfg.ref_trial_len) == 0
        cleared = ~self.food_alive.any(-1)
        which = cleared | torch.full_like(cleared, bool(due))
        if not bool(which.any()):
            return
        self._spawn_referential(which)
        start = self._t(cfg.ref_start_xy)
        jit = (torch.rand((self.B, self.F - 1, 2), generator=self.g, device=self.dev,
                          dtype=self.dt_type) - 0.5) * 4.0
        self.pos[:, 1:] = torch.where(which[:, None, None], start + jit, self.pos[:, 1:])
        self.theta[:, 1:] = torch.where(
            which[:, None], torch.full_like(self.theta[:, 1:], math.pi / 2), self.theta[:, 1:])
        # Re-baseline the shaping reference at the teleport.  Without this the
        # jump back to the gate registers as a large increase in distance-to-site
        # and the agent is punished precisely for finishing a trial.
        sites = self._t(cfg.ref_sites)
        d_new = torch.cdist(self.pos, sites[None].expand(self.B, 2, 2)).min(-1).values
        self.prev_food_dist = torch.where(which[:, None], d_new, self.prev_food_dist)
        self._arrived = torch.where(which[:, None], torch.zeros_like(self._arrived), self._arrived)
        newbit = (torch.rand((self.B,), generator=self.g, device=self.dev) < 0.5)
        self._rand_bit = torch.where(which, newbit, self._rand_bit)
        shadow = (torch.rand((self.B,), generator=self.g, device=self.dev) < 0.5).long()
        self._active_msg = torch.where(which, shadow, self._active_msg)

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------
    def _sense_mormyromast(self, emit_self: torch.Tensor, emit_illum: torch.Tensor | None = None) -> torch.Tensor:
        """Active electrolocation: object images carried by EOD fields.

        For each sensing fish i a separate scene is built whose active emitters
        are given by a per-receiver mask: fish i itself if `emit_self[i]`, and
        each conspecific j if `emit_illum[j]`.  Every non-emitting fish and every
        food item is an insulating sphere polarised by that total field, and the
        receptor reads the field of the induced dipoles alone.

        Splitting the mask this way is what makes the *cue* arm of the
        cue-versus-signal question physically real: a conspecific's pulse
        illuminates the receiver's world whether or not the emitter gains
        anything by it, exactly the situation in which a receiver "merely
        exploits" someone else's probe (Pedraja & Sawtell 2024).
        """
        cfg = self.cfg
        B, F = self.B, self.F
        Nf = cfg.n_food
        BF = B * F
        mode = cfg.collective_sensing
        if emit_illum is None:
            emit_illum = emit_self

        eyeF = torch.eye(F, device=self.dev, dtype=self.dt_type)[None]            # (1,F,F)
        es = emit_self.to(self.dt_type)[:, :, None] * eyeF                         # own pulse
        ei = emit_illum.to(self.dt_type)[:, None, :] * (1 - eyeF)                  # neighbours' pulses
        if mode == 0:
            mask = es                       # self-image only
        elif mode == 2:
            mask = ei                       # conspecific image only
        else:
            mask = es + ei                  # both (upstream default)

        # --- monopole sources, masked per receiver ------------------------
        mono_w = P.to_world(self.mono_ego[None, None], self.theta, self.pos)       # (B,F,2,2)
        mono_all = mono_w[:, None].expand(B, F, F, 2, 2).reshape(BF, F * 2, 2)
        q = (self.mono_q[None, None] * mask[..., None]).reshape(BF, F * 2)

        # --- conductors: food + fish that are silent in this scene --------
        cond_pos = torch.cat([self.food, self.pos], dim=1)                         # (B,Nf+F,2)
        cond_pos = cond_pos[:, None].expand(B, F, Nf + F, 2).reshape(BF, Nf + F, 2)

        food_contrast = C.FOOD_CONTRAST * self.food_alive.to(self.dt_type)         # (B,Nf)
        food_contrast = food_contrast[:, None].expand(B, F, Nf).reshape(BF, Nf)
        # a fish that is discharging in this scene does not also act as a conductor
        fish_contrast = C.FISH_CONTRAST * (1.0 - (mask > 0).to(self.dt_type)).reshape(BF, F)
        contrast = torch.cat([food_contrast, fish_contrast], dim=1)

        radius = torch.cat(
            [
                torch.full((B, Nf), C.FOOD_RADIUS_CM, device=self.dev, dtype=self.dt_type),
                torch.full((B, F), C.BODY_RADIUS_CM, device=self.dev, dtype=self.dt_type),
            ],
            dim=1,
        )[:, None].expand(B, F, Nf + F).reshape(BF, Nf + F)

        e_at_cond = P.field_from_monopoles(cond_pos, mono_all, q)
        moments = P.induce_dipoles(e_at_cond, contrast, radius)
        moments = torch.cat(
            [moments[:, :Nf], P.clip_moments(moments[:, Nf:], radius[:, Nf:], C.MAX_CHARGE_ALLOWED)],
            dim=1,
        )

        dp, dm = P.with_images_dip(cond_pos, moments, cfg.arena_cm, cfg.use_reflections)
        s_pos = P.to_world(self.morm_pos_ego[None, None], self.theta, self.pos).reshape(BF, C.NUM_MORM, 2)
        s_nrm = P.rotate(self.morm_nrm_ego[None, None], self.theta).reshape(BF, C.NUM_MORM, 2)

        if mode == 0:
            # Legacy path: subtract the fixed corollary-discharge template rather
            # than the instantaneous EOD field.  Retained because it is the
            # configuration validated bit-for-bit against upstream.
            mp, mq = P.with_images_mono(mono_all, q, cfg.arena_cm, cfg.use_reflections)
            e = P.measure_field(s_pos, mp, mq, dp, dm)
            raw = P.project(e, s_nrm).reshape(B, F, C.NUM_MORM)
            raw = raw - self.morm_cd[None, None] * emit_self[..., None].to(self.dt_type)
        else:
            # Dynamic baseline: the direct EOD field is cancelled exactly, so the
            # residual is the object image alone.
            e = P.measure_field(s_pos, None, None, dp, dm)
            raw = P.project(e, s_nrm).reshape(B, F, C.NUM_MORM)
            # images borrowed from a neighbour's discharge are far weaker
            cons_only = (~emit_self).to(self.dt_type)[..., None]
            raw = raw * (1 + cons_only * (C.MORM_CONS_MULTIPLIER - 1))

        if cfg.noise_frac_morm > 0:
            n = cfg.noise_frac_morm
            raw = raw * (1 + (torch.rand(raw.shape, generator=self.g, device=self.dev, dtype=self.dt_type) * 2 - 1) * n)
        out = P.process_morm(raw)
        return out if cfg.morm_enabled else torch.zeros_like(out)

    def _sense_knollen(self, emit_social: torch.Tensor, slot_perm: torch.Tensor | None = None):
        """Knollenorgan array: detection of *conspecific* discharges.

        Returns (obs, meta, detect) where obs is (B,F,(F-1)*K) with the emitter
        held fixed per slot, so sender identity is preserved, and `detect` is a
        (B,F,F-1) indicator of which senders were heard at all.
        """
        cfg = self.cfg
        B, F, K = self.B, self.F, C.NUM_KNOLLEN
        BF = B * F

        mono_w = P.to_world(self.mono_ego[None, None], self.theta, self.pos)  # (B,F,2,2)
        s_pos = P.to_world(self.kno_pos_ego[None, None], self.theta, self.pos)  # (B,F,K,2)
        s_nrm = P.rotate(self.kno_nrm_ego[None, None], self.theta)

        # scene e: only emitter e discharges; measure at every fish's array
        mono_e = mono_w.reshape(BF, 2, 2)
        q_e = self.mono_q[None].expand(BF, 2) * emit_social.reshape(BF, 1).to(self.dt_type)
        query = s_pos.reshape(B, F * K, 2)[:, None].expand(B, F, F * K, 2).reshape(BF, F * K, 2)
        # knollen omits wall reflections (upstream convention)
        e = P.field_from_monopoles(query, mono_e, q_e)
        nrm = s_nrm.reshape(B, F * K, 2)[:, None].expand(B, F, F * K, 2).reshape(BF, F * K, 2)
        raw = P.project(e, nrm).reshape(B, F, F, K)  # (B, emitter, receiver, K)
        raw = raw.permute(0, 2, 1, 3)               # (B, receiver, emitter, K)

        raw = raw * cfg.knollen_gain
        if cfg.noise_frac_knollen > 0:
            n = cfg.noise_frac_knollen
            raw = raw * (1 + (torch.rand(raw.shape, generator=self.g, device=self.dev, dtype=self.dt_type) * 2 - 1) * n)

        binar = P.binarize(raw, C.KNOLLEN_BINARIZE_THRESHOLD, C.KNOLLEN_SENSOR_MIN)

        # drop the self slot: (B, F, F-1, K)
        idx = self._offdiag_index()  # (F, F-1)
        binar = binar.gather(2, idx[None, :, :, None].expand(B, F, F - 1, K))

        # sender-size metadata (a dominance cue carried by the pulse)
        size_pairs = self.size[:, None, :].expand(B, F, F).gather(2, idx[None].expand(B, F, F - 1))
        rel = size_pairs - self.size[..., None]
        if cfg.noise_frac_knollen_meta > 0:
            n = cfg.noise_frac_knollen_meta
            rel = rel + (torch.rand(rel.shape, generator=self.g, device=self.dev, dtype=self.dt_type) * 2 - 1) * n
        heard = (binar.abs().sum(-1) > 0).to(self.dt_type)          # (B,F,F-1)
        meta = (rel.clamp(-1, 1)) * heard

        if slot_perm is not None:
            binar = torch.gather(binar, 2, slot_perm[..., None].expand(B, F, F - 1, K))
            meta = torch.gather(meta, 2, slot_perm)
            heard = torch.gather(heard, 2, slot_perm)

        if not cfg.knollen_enabled:
            binar = torch.zeros_like(binar)
            meta = torch.zeros_like(meta)
            heard = torch.zeros_like(heard)
        return binar.reshape(B, F, (F - 1) * K), meta, heard

    def _offdiag_index(self):
        if not hasattr(self, "_odi"):
            F = self.F
            rows = []
            for i in range(F):
                rows.append([j for j in range(F) if j != i])
            self._odi = torch.tensor(rows, device=self.dev, dtype=torch.long)
        return self._odi

    def _sense_ampullary(self, emit_social: torch.Tensor | None = None) -> torch.Tensor:
        """Passive DC channel: the standing bioelectric fields of prey, fish and predators.

        Independent of any discharge, so it provides a non-signal route to social
        information and keeps the knollen channel from being the only way to know
        that a conspecific exists.  A conspecific discharge degrades it (upstream
        `noise_frac_amp_cons_eod`), so emitting imposes a sensory cost on
        neighbours -- the simplest form of interference competition available to
        an electric fish.
        """
        cfg = self.cfg
        B, F, Nf = self.B, self.F, cfg.n_food

        food_mom = P.rotate_one(self.food_intrinsic_ego.expand(B, Nf, 2), self.food_theta)
        food_mom = food_mom * self.food_alive[..., None].to(self.dt_type)
        fish_mom = P.rotate_one(self.fish_intrinsic_ego.expand(B, F, 2), self.theta)
        dip_pos = [self.food, self.pos]
        dip_mom = [food_mom, fish_mom]
        if cfg.predation > 0 and cfg.n_pred > 0:
            pm = self.pred_intrinsic_ego.expand(B, cfg.n_pred, 2)
            dip_pos.append(self.pred_pos)
            dip_mom.append(pm)
        dp, dm = P.with_images_dip(
            torch.cat(dip_pos, dim=1), torch.cat(dip_mom, dim=1), cfg.arena_cm, cfg.use_reflections
        )

        s_pos = P.to_world(self.amp_pos_ego[None, None], self.theta, self.pos).reshape(B, F * C.NUM_AMP, 2)
        s_nrm = P.rotate(self.amp_nrm_ego[None, None], self.theta).reshape(B, F * C.NUM_AMP, 2)
        e = P.measure_field(s_pos, None, None, dp, dm)
        raw = P.project(e, s_nrm).reshape(B, F, C.NUM_AMP)
        raw = raw - self.amp_intrinsic_baseline[None, None]

        n = torch.full((B, F), cfg.noise_frac_amp, device=self.dev, dtype=self.dt_type)
        if emit_social is not None and cfg.noise_frac_amp_cons_eod > 0:
            others = emit_social.to(self.dt_type).sum(1, keepdim=True) - emit_social.to(self.dt_type)
            n = torch.where(others > 0, torch.full_like(n, cfg.noise_frac_amp_cons_eod), n)
        u = torch.rand(raw.shape, generator=self.g, device=self.dev, dtype=self.dt_type) * 2 - 1
        raw = raw * (1 + u * n[..., None])
        out = P.process_sensor_readings(raw, C.AMP_SENSOR_MIN, C.AMP_SENSOR_MAX)
        return out if cfg.amp_enabled else torch.zeros_like(out)

    def _observe(self, emit_self, emit_social, slot_perm=None, emit_illum=None):
        if emit_illum is None:
            emit_illum = emit_self if self.cfg.illuminate_others else torch.zeros_like(emit_self)
        morm = self._sense_mormyromast(emit_self, emit_illum)
        kno, meta, heard = self._sense_knollen(emit_social, slot_perm)
        extra = []
        if self.cfg.signal_channel:
            idx = self._offdiag_index()
            sg = getattr(self, "_sig", None)
            if sg is None:
                sg = torch.zeros((self.B, self.F), device=self.dev, dtype=torch.bool)
            if self.cfg.yoked_knollen and hasattr(self, "_yoke_perm"):
                sg = sg[self._yoke_perm]
            sgf = (sg.to(self.dt_type) * 2 - 1)[:, None, :].expand(self.B, self.F, self.F)
            sgf = torch.gather(sgf, 2, idx[None].expand(self.B, self.F, self.F - 1))
            if slot_perm is not None:
                sgf = torch.gather(sgf, 2, slot_perm)
            sgf = sgf * heard             # audible only when the pulse was heard
            # eval-time ablations of the decoupled variable, leaving the pulse
            # (and therefore all sensing) completely untouched
            if getattr(self, "_kill_subtype", False):
                sgf = torch.zeros_like(sgf)
            elif getattr(self, "_scramble_subtype", False):
                sgf = sgf[torch.randperm(self.B, generator=self.g, device=self.dev)]
            extra.append(sgf)
        if self.cfg.task == "referential":
            cue = torch.zeros((self.B, self.F), device=self.dev, dtype=self.dt_type)
            cue[:, 0] = self.active.to(self.dt_type) * 2 - 1   # sender only
            extra.append(cue[..., None])
            # Egocentric bearing to each candidate site.  The fish knows where
            # the two patches are; what it does not know is which holds food.
            # Supplying this makes the task a signalling problem rather than a
            # navigation problem, which is what we mean to be studying.
            sites = self._t(self.cfg.ref_sites)                    # (2,2)
            rel = sites[None, None] - self.pos[:, :, None, :]      # (B,F,2,2)
            ang = torch.atan2(rel[..., 1], rel[..., 0]) - self.theta[..., None]
            extra.append(torch.cos(ang))
            extra.append(torch.sin(ang))
        amp = self._sense_ampullary(emit_social)
        self._last_heard = heard
        obs = torch.cat(
            [
                morm,
                amp,
                kno,
                meta,
                torch.tanh(self.last_act),
                self.was_bitten[..., None],
                self.size[..., None],
                self.bite_cd[..., None],
                self.eat_cd[..., None],
                *extra,
            ],
            dim=-1,
        )
        return obs

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor, channel: ChannelSpec | None = None):
        """action: (B, F, 4) in [-1,1] = (thrust, turn, emit, bite)."""
        cfg = self.cfg
        B, F = self.B, self.F
        w, h = cfg.arena_cm
        action = action.clamp(-1, 1)
        if cfg.task == "referential":
            action = action.clone()
            action[:, 0, :2] = 0.0                      # the sender cannot move
            if cfg.ref_sender_emits:
                action[:, 0, 2] = 1.0                   # nor choose when to pulse
            if cfg.ref_scripted != "none" and cfg.signal_channel:
                if cfg.ref_scripted == "honest":
                    ref = self._active_msg if cfg.ref_mediate else self.active
                    bit = ref.to(self.dt_type)
                else:
                    # 'random': a bit that is constant within a trial, exactly
                    # like the honest one, but uncorrelated with the truth --
                    # matched in every respect except informativeness
                    bit = self._rand_bit.to(self.dt_type)
                action[:, 0, 4] = bit * 2 - 1
        emit = (action[..., 2] > 0) & cfg.eod_allowed
        bite = action[..., 3] > 0
        sig = (action[..., 4] > 0) if cfg.signal_channel else torch.zeros_like(emit)
        self._sig = sig & emit          # the subtype rides on an actual discharge

        emit_self, emit_illum, emit_social, slot_perm = self._apply_channel(emit, channel)
        if cfg.scramble_id_always and slot_perm is None:
            slot_perm = torch.argsort(
                torch.rand((B, F, F - 1), generator=self.g, device=self.dev), dim=-1
            )
        self._last_emit_self = emit_self
        self._last_emit_social = emit_social

        # --- locomotion (first-order velocity control, size-scaled) ------
        size_mult = (1.0 + self.size)
        omega = action[..., 1] * C.MAX_ANGULAR_VELOCITY * size_mult
        frozen = (self.eat_cd > 0).to(self.dt_type)
        omega = omega * (1 - frozen)
        self.theta = torch.atan2(torch.sin(self.theta + omega), torch.cos(self.theta + omega))
        speed = action[..., 0].clamp(min=0) * C.MAX_LINEAR_VELOCITY * size_mult * (1 - frozen)
        heading = torch.stack([torch.cos(self.theta), torch.sin(self.theta)], -1)
        newpos = self.pos + heading * speed[..., None]

        # Agent-agent collision blocks the move (walls merely clamp, unpenalised,
        # matching upstream `apply_movement_step`).
        d_prop = torch.cdist(newpos, self.pos)
        eye_f = torch.eye(F, device=self.dev, dtype=torch.bool)[None]
        collided = ((d_prop < 2 * C.BODY_RADIUS_CM) & ~eye_f).any(-1)
        newpos = torch.where(collided[..., None], self.pos, newpos)
        newpos[..., 0] = newpos[..., 0].clamp(C.BODY_RADIUS_CM, w - C.BODY_RADIUS_CM)
        newpos[..., 1] = newpos[..., 1].clamp(C.BODY_RADIUS_CM, h - C.BODY_RADIUS_CM)
        self.pos = newpos

        # --- foraging ----------------------------------------------------
        d_food = torch.cdist(self.pos, self.food)                       # (B,F,Nf)
        alive = self.food_alive[:, None, :]
        d_food_masked = torch.where(alive, d_food, torch.full_like(d_food, 1e6))

        to_food = self.food[:, None, :, :] - self.pos[:, :, None, :]
        ang = torch.atan2(to_food[..., 1], to_food[..., 0]) - self.theta[..., None]
        ang = torch.atan2(torch.sin(ang), torch.cos(ang))
        in_cone = ang.abs() < C.EATING_ANGLE
        can_eat = (d_food_masked < C.EATING_RADIUS_CM) & in_cone & (self.eat_cd <= 0)[..., None]

        # resolve contested items: the largest fish wins
        prio = self.size[:, :, None] * can_eat.to(self.dt_type) - (~can_eat).to(self.dt_type)
        winner = prio.argmax(dim=1)                                      # (B,Nf)
        onehot = torch.zeros_like(can_eat)
        onehot.scatter_(1, winner[:, None, :], True)
        ate = can_eat & onehot & alive
        n_ate = ate.sum(-1).to(self.dt_type)                             # (B,F)
        if cfg.food_exclusive:
            self.food_alive = self.food_alive & ~ate.any(1)
        self.eat_cd = torch.where(n_ate > 0, torch.full_like(self.eat_cd, C.EAT_COOLDOWN_STEPS), (self.eat_cd - 1).clamp(min=0))

        # --- aggression --------------------------------------------------
        d_fish = torch.cdist(self.pos, self.pos)
        eye = torch.eye(F, device=self.dev, dtype=torch.bool)[None]
        to_f = self.pos[:, None, :, :] - self.pos[:, :, None, :]
        angf = torch.atan2(to_f[..., 1], to_f[..., 0]) - self.theta[..., None]
        angf = torch.atan2(torch.sin(angf), torch.cos(angf))
        can_bite = (
            (d_fish < C.BITING_RADIUS_CM) & ~eye & (angf.abs() < C.EATING_ANGLE)
            & bite[..., None] & (self.bite_cd <= 0)[..., None]
        )
        self.was_bitten = can_bite.any(1).to(self.dt_type)
        bit_other = can_bite.any(-1).to(self.dt_type)
        self.bite_cd = torch.where(bit_other > 0, torch.full_like(self.bite_cd, C.BITE_COOLDOWN_STEPS), (self.bite_cd - 1).clamp(min=0))
        # size difference of the aggressor, for the scaled bite penalty
        biter_size = (can_bite.to(self.dt_type) * self.size[:, :, None]).sum(1)
        n_biters = can_bite.to(self.dt_type).sum(1).clamp(min=1)
        biter_size = biter_size / n_biters

        # --- predation ---------------------------------------------------
        struck = self._step_predators(emit_self)

        # --- reward ------------------------------------------------------
        r = torch.zeros((B, F), device=self.dev, dtype=self.dt_type)
        r = r + cfg.r_eat * n_ate
        if cfg.shared_food > 0:
            group = n_ate.sum(1, keepdim=True) - n_ate
            r = r + cfg.r_eat * cfg.shared_food * group / max(F - 1, 1)

        if cfg.task == "referential":
            # distance to the nearest *site*, which is the same information for
            # both sites and therefore cannot reveal which one holds food
            sites_w = self._t(cfg.ref_sites)
            nearest = torch.cdist(self.pos, sites_w[None].expand(B, 2, 2)).min(-1).values
            shape_coef = cfg.ref_site_shaping
        else:
            nearest = d_food_masked.min(-1).values
            nearest = torch.where(nearest > 1e5, torch.full_like(nearest, float("nan")), nearest)
            shape_coef = cfg.r_shaping
        prev = self.prev_food_dist
        shaping = torch.where(
            torch.isnan(prev) | torch.isnan(nearest),
            torch.zeros_like(nearest),
            shape_coef * (prev - nearest) / (w + h),
        )
        r = r + shaping
        self.prev_food_dist = nearest

        r = r + cfg.r_bitten * self.was_bitten * (1 + (biter_size - self.size))
        r = r + cfg.r_bite * bit_other
        r = r + cfg.r_collision * collided.to(self.dt_type)
        info_arrived = None
        if cfg.task == "referential" and cfg.ref_arrival_reward:
            # Score the referential task on reaching the site the signal names,
            # once per trial.  Scoring it on food eaten would confound signal use
            # with close-range foraging skill, which is not what is being tested.
            sites_a = self._t(cfg.ref_sites)
            d_all = torch.cdist(self.pos, sites_a[None].expand(B, 2, 2))     # (B,F,2)
            d_act = torch.gather(d_all, 2, self.active[:, None, None].expand(B, F, 1))[..., 0]
            near = d_act < cfg.ref_at_site_cm
            hit = near & (~self._arrived)
            r = r + cfg.ref_arrival_reward * hit.to(self.dt_type)
            self._arrived = self._arrived | near
            info_arrived = hit
        r = r - cfg.eod_cost * emit_self.to(self.dt_type)
        if cfg.signal_channel and cfg.signal_cost:
            r = r - cfg.signal_cost * self._sig.to(self.dt_type)
        r = r - cfg.predation * struck

        self.last_act = action[..., :4]
        self.t += 1
        if cfg.task == "referential":
            self._referential_trial_reset()
        obs = self._observe(emit_self, emit_social, slot_perm, emit_illum)
        done = self.t >= cfg.episode_len
        info = {
            "emit": emit,
            "emit_self": emit_self,
            "signal": self._sig,
            "emit_illum": emit_illum,
            "emit_social": emit_social,
            "ate": n_ate,
            "struck": struck,
            "heard": self._last_heard,
            "bit": bit_other,
            "bitten": self.was_bitten,
            "nearest_food": nearest,
        }
        if cfg.task == "referential":
            sites = self._t(cfg.ref_sites)
            d_site = torch.cdist(self.pos, sites[None].expand(B, 2, 2))  # (B,F,2)
            info["active"] = self.active
            info["chose"] = d_site.argmin(-1)
            info["at_site"] = (d_site.min(-1).values < cfg.ref_at_site_cm)
            info["arrived"] = (info_arrived if info_arrived is not None
                               else torch.zeros_like(self._arrived))
        return obs, r, done, info

    def _step_predators(self, emit_self: torch.Tensor) -> torch.Tensor:
        """Electroreceptive predators that localise prey by their discharges.

        A predator only perceives fish that discharged this step and are within
        `pred_detect_cm`; it then advances toward the loudest such source.  A
        silent fish is invisible, but a silent fish near a discharging
        neighbour is still exposed -- signalling imposes a risk externality on
        the group.
        """
        cfg = self.cfg
        B, F = self.B, self.F
        struck = torch.zeros((B, F), device=self.dev, dtype=self.dt_type)
        if cfg.predation <= 0 or cfg.n_pred <= 0:
            return struck
        w, h = cfg.arena_cm

        d = torch.cdist(self.pred_pos, self.pos)                       # (B,P,F)
        audible = emit_self[:, None, :] & (d < cfg.pred_detect_cm)
        # apparent loudness falls off with distance
        loud = torch.where(audible, 1.0 / (d + 1.0), torch.zeros_like(d))
        has = loud.sum(-1) > 0                                          # (B,P)
        tgt = loud.argmax(-1)                                           # (B,P)
        tgt_pos = torch.gather(self.pos, 1, tgt[..., None].expand(B, cfg.n_pred, 2))
        direction = tgt_pos - self.pred_pos
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        active = has & (self.pred_cd <= 0)
        self.pred_pos = self.pred_pos + direction * cfg.pred_speed_cm * active[..., None].to(self.dt_type)
        self.pred_pos[..., 0] = self.pred_pos[..., 0].clamp(0, w)
        self.pred_pos[..., 1] = self.pred_pos[..., 1].clamp(0, h)

        d2 = torch.cdist(self.pred_pos, self.pos)
        hit = (d2 < cfg.pred_strike_cm) & (self.pred_cd <= 0)[..., None]
        struck = hit.any(1).to(self.dt_type)
        fired = hit.any(-1)
        self.pred_cd = torch.where(fired, torch.full_like(self.pred_cd, cfg.pred_cooldown), (self.pred_cd - 1).clamp(min=0))
        # respawn a predator that has just struck, so pressure stays stationary
        resp = fired[..., None].to(self.dt_type)
        rnd = torch.rand(self.pred_pos.shape, generator=self.g, device=self.dev, dtype=self.dt_type)
        rnd[..., 0] *= w
        rnd[..., 1] *= h
        self.pred_pos = self.pred_pos * (1 - resp) + rnd * resp
        return struck

    # ------------------------------------------------------------------
    # Counterfactual channels
    # ------------------------------------------------------------------
    def _apply_channel(self, emit: torch.Tensor, spec: ChannelSpec | None):
        """Split one motor act into its private and public consequences."""
        B, F = self.B, self.F
        illum_default = emit if self.cfg.illuminate_others else torch.zeros_like(emit)
        if self.cfg.yoked_knollen:
            # what each receiver hears comes from another arena; the geometry
            # (bearing, distance) is still its own, only the contingency is cut
            perm = torch.randperm(self.B, generator=self.g, device=self.dev)
            self._yoke_perm = perm
            heard_mask = emit[perm]
            if spec is None or spec.mode == "intact":
                return emit, illum_default, heard_mask, None
        if spec is None or spec.mode == "intact":
            return emit, illum_default, emit, None

        sel = torch.zeros((F,), device=self.dev, dtype=torch.bool)
        if spec.agents is None:
            sel[:] = True
        else:
            sel[list(spec.agents)] = True
        sel = sel[None].expand(B, F)

        emit_self, emit_illum, emit_social, slot_perm = emit, illum_default, emit, None
        m = spec.mode
        if m == "mute":
            # the upstream intervention: every consequence removed at once
            emit_self = emit & ~sel
            emit_illum = illum_default & ~sel
            emit_social = emit & ~sel
        elif m == "private":
            emit_illum = illum_default & ~sel
            emit_social = emit & ~sel
        elif m == "cue_only":
            emit_social = emit & ~sel
        elif m == "signal_only":
            emit_illum = illum_default & ~sel
        elif m == "social":
            emit_self = emit & ~sel
        elif m == "phantom":
            ph = torch.rand((B, F), generator=self.g, device=self.dev) < spec.phantom_rate
            emit_social = torch.where(sel, ph, emit)
        elif m == "replay":
            if spec.replay_train is None:
                raise ValueError("replay mode needs `replay_train`")
            tr = spec.replay_train
            idx = self.t % tr.shape[1]
            emit_social = torch.where(sel, tr[:, idx].to(self.dev), emit)
        elif m == "scramble_time":
            if spec.replay_train is None:
                raise ValueError("scramble_time needs a shuffled `replay_train`")
            tr = spec.replay_train
            idx = self.t % tr.shape[1]
            emit_social = torch.where(sel, tr[:, idx].to(self.dev), emit)
        elif m == "scramble_id":
            slot_perm = torch.argsort(
                torch.rand((B, F, F - 1), generator=self.g, device=self.dev), dim=-1
            )
        else:
            raise ValueError(f"unknown channel mode {m}")
        return emit_self, emit_illum, emit_social, slot_perm
