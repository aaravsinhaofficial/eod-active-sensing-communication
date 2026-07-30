# Communication is a mediated effect

Separating signals from exploited cues in dual-use action channels, with an
application to active electrosensing.

An action that both senses and broadcasts is hard to classify. A weakly electric
fish's discharge illuminates its own surroundings *and* announces its presence to
anyone with the receptors to detect it. The same structure arises for active
sonar, a robot's laser sweep, or any observable act of sensing.

## The problem with the standard diagnostics

Three measures are in common use, and **each is individually foolable**. We show
this on dyads whose communication status is known by construction — a Lewis
signalling game embedded in the simulator, with an immobile sender, a forced pulse
schedule so timing carries nothing, and a one-bit subtype as the only free channel.

| dyad | positive signalling | causal influence | payoff of deletion | **NIE (ours)** |
|---|---|---|---|---|
| informative + attended | 0.934 | 31.4 | −173 | **−95.5** |
| informative + ignored | 0.934 ✗ | 0.000 | −0.08 | −0.90 |
| noise + attended | 0.0002 | 31.3 ✗ | −75.5 ✗ | −0.27 |
| noise + ignored | 0.0002 | 0.000 | +0.78 | −0.32 |

Positive signalling is satisfied by a sender nobody listens to. Causal influence
is satisfied by a receiver attending to noise. Ablation fires for a channel
carrying nothing — because deleting a message takes the receiver *outside its own
training distribution*.

## The measure

Communication is a **natural indirect effect**: the sender's private state acting
on the receiver's payoff *through* the message.

> NIE = E[ u_R(s, M(s')) − u_R(s, M(s)) ],  s' ~ p(S) drawn independently

Hold the world at `s`, so every direct pathway is untouched, and transmit the
message the sender *would have* produced in an independently drawn world `s'`.
Because `M(s')` comes from the message's own marginal, the receiver never leaves
distribution — the only thing broken is the correspondence between message and
world.

We prove it is zero whenever the message is independent of the sender's state
(Prop. 1) or the receiver ignores it (Prop. 2), and that deletion is *not*
equivalent (Prop. 3, with the counterexample realised above). It is the only one
of the four measures that answers all four validation dyads correctly.

## Applied to the electric discharge

- **Muting is mostly blinding.** Silencing a fish costs it food; the overwhelming
  majority is its own lost electrolocation, not lost detectability by others. The
  private share holds at 83–135% across 2/4/6 fish, a 100 cm arena, 1024-step
  episodes and a 256-unit GRU.
- **The channel passes both standard criteria** — decodable food information
  (ΔR² = 0.156), receiver-policy influence 13× the noise floor — **yet its
  mediated effect is 0.49% of receiver return, CI spanning zero**, against a
  demonstrated detection floor of −95.5. An exploited cue, not a signal.
- **Two diagnostics fail outright here too.** Positive signalling is *higher* in
  deaf populations (0.152 vs 0.086); causal influence *rises* 12-fold as the
  channel becomes less informative.
- **Cost does not manufacture semantics.** Cost and eavesdropping predation drive
  rate down and crypsis works, but the train becomes *less* informative per pulse.
- **It is about dual use, not fish.** A minimal benchmark with no fish physics
  reproduces the decomposition: 100% private share, detection share −0.06.

## An honest bound on the learner

Recurrent MAPPO does not solve the referential game in **36 runs at 50M steps** —
including with a scripted honest sender, and with the Eccles et al.
positive-signalling bias — while a hand-coded dyad gets 4.00/4 from the same
observations. Lower action noise and a halved credit horizon do not help. The
measurement results stand independent of this; the "cost/decoupling does not
create signalling" claims are scoped to a standard learner and say so.

## What is here

```
eodcomm/
  constants.py      physical + morphological constants, transcribed from the reference simulator
  physics.py        batched electrostatics on GPU (Coulomb monopole/dipole, method of images,
                    induced dipoles, receptor transduction)
  env.py            the batched multi-agent environment: three electrosensory channels,
                    the channel algebra, metabolic cost, EOD-guided predators,
                    cooperative/competitive harvests, tunable social range
  ppo.py            recurrent MAPPO with a centralised critic and a hybrid action space
  metrics.py        interventional causal influence, positive signalling, positive listening,
                    receiver-side content decoding, Sender Shaping Index
  interventions.py  the counterfactual channel battery (private / cue-only / signal-only /
                    phantom / replay / scramble) with paired-within-arena bootstrap statistics
  train.py          training driver
  plotstyle.py      figure style (CVD-validated categorical palette)
scripts/
  validate_physics.py  numerical agreement against the reference NumPy implementation
  launch_all.py        defines and launches the full experimental programme
  analyze.py           evaluates every checkpoint into a metric record
  figures.py           builds every figure and the paper's numeric macros
paper/               LaTeX source, bibliography, compiled PDF
results/             validation report, per-run metrics, figures
```

## Reproducing

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy matplotlib pandas scikit-learn

# 1. verify the physics reproduces the reference implementation
git clone https://github.com/KempnerInstitute/wef.git ../wef_upstream
python3 scripts/validate_physics.py

# 2. train every condition (270 runs total; ~3 h on 2 GPUs)
python3 scripts/launch_all.py        # channel decomposition + phase grid
python3 scripts/launch_decoupled.py  # decoupled positive control + yoked SSI control

# 3. evaluate and plot
bash scripts/run_analysis.sh
python3 scripts/figures.py

# 4. build the paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Validation

`scripts/validate_physics.py` checks every stage of the sensory pipeline against
the reference NumPy code in double precision on random scenes. All checks agree to
floating-point round-off; the knollenorgan and ampullary pathways agree
bit-for-bit. The report is written to `results/physics_validation.json`.

One finding from that exercise is worth flagging for anyone using the reference
code: `cfg.ELECTRIC_CONSTANTS["reflection_scale"] = 0.95`, but
`ElectricScene._build_reflections` never forwards it to `reflect_sources`, whose
default is `1.0`. The published results therefore use unattenuated wall images.
We match the code rather than the configuration file.

## Relation to the reference simulator

This is an independent reimplementation of the environment physics of
[KempnerInstitute/wef](https://github.com/KempnerInstitute/wef), rewritten to run
thousands of arenas simultaneously on GPU (~5.8×10⁵ agent-steps/s aggregate on two
RTX PRO 6000) so that a factorial sweep over cost, predation, economics, range and
identity is affordable. Constants, receptor layouts, transduction functions and
reward coefficients are taken from that codebase. Deliberate divergences are
listed in the paper's Methods §"Deliberate divergences".

## Licence

MIT for the code in `eodcomm/` and `scripts/`. The reference simulator it is
validated against is separately licensed by its authors.
