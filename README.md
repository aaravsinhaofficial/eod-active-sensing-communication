# When does active sensing become communication?

Code and results for a study of when an electric organ discharge (EOD) — a pulse
a weakly electric fish produces to *see* — becomes a *signal*.

Every EOD does three things at once:

| consequence | who benefits | what it is |
|---|---|---|
| **reafference** — the pulse casts an electric image on the emitter's own mormyromasts | emitter | active electrolocation |
| **illumination** — the same field casts images on a *neighbour's* mormyromasts | neighbour | an exploitable **cue** |
| **detection** — the pulse fires the neighbour's knollenorgans | neighbour | a candidate **signal** |

Silencing a fish — the standard intervention, and the one used by
[Singh, Johnson-Yu et al. (2025)](https://arxiv.org/abs/2511.08436) — removes all
three at once, so the resulting behavioural change cannot be attributed to any of
them. This repository makes the three independently switchable, and adds a fourth
condition in which a signalling variable is *decoupled* from sensing altogether.

## Headline findings

### 1. The assays are validated before they are trusted

A Lewis signalling game is embedded in the same physics: an immobile sender
privately observes which of two sites holds food, its discharges are forced onto a
fixed schedule so timing carries nothing, and a one-bit discharge subtype is the
only free channel. Crossing an informative against an uninformative sender, and an
attentive against a deaf receiver, gives four dyads of known status.

| dyad | task success | content MI | causal influence | payoff of deleting |
|---|---|---|---|---|
| informative + attended | **4.00/4** | **0.932** | 31.5 | **-172.5** |
| informative + ignored | 1.96/4 | 0.932 | **0.000** | -1.3 n.s. |
| noise + attended | 1.93/4 | **0.001** | 31.9 | -72.8 |
| noise + ignored | 1.89/4 | 0.001 | 0.000 | -0.2 n.s. |

**Every diagnostic alone is foolable.** Positive signalling is identical whether
anyone listens. Causal influence is identical whether the channel carries
anything. Even the payoff-ablation test fires at -72.8 for pure noise the receiver
has organised around. Only the conjunction identifies communication.

### 2. Applied to the electric discharge

- **Muting is mostly blinding.** Silencing a fish costs it 0.52 items/episode;
  0.51 is its own lost electrolocation, 0.03 the lost detectability by others. The
  private share holds at 83-135% across 2/4/6 fish, a 100 cm arena, 1024-step
  episodes and a 256-unit GRU.
- **It passes both standard criteria and still is not communicating.** Decodable
  food information (dR2 = 0.156) and receiver-policy influence 13x the noise floor
  - yet replaying, scrambling or fabricating the channel moves no payoff we can
  resolve.
- **Two diagnostics fail outright.** Positive signalling is *higher* in deaf
  populations (0.152 vs 0.086). Causal influence *rises* 12-fold as the channel
  becomes less informative.
- **Sender shaping needs the right control.** Disabling knollenorgans removes
  *reception* and measures free-riding (hearing fish emit less, 0.545 vs 0.611).
  Holding reception fixed and cutting only audibility gives 0.040 - a cue.
- **Cost does not manufacture semantics.** Cost and eavesdropping predation drive
  rate down monotonically and crypsis works, but the pulse train becomes *less*
  informative, even per pulse.
- **Freeing the channel is not sufficient.** A decoupled subtype is used at chance
  (0.52) and deleting it changes nobody's payoff in any of four settings (12 seeds
  each, no cell surviving correction).

### 3. It is about dual use, not about fish

A minimal benchmark with no fish physics - agents forage in a unit square, a
`ping` shows the pinger nearby items, shows neighbours *their* neighbourhood, and
is registered as an event - reproduces the decomposition: muting costs an agent
9.89 items, **100% of it private reafference**, detection share -0.06.

### 4. The optimiser is a binding constraint (stated, not hidden)

Recurrent MAPPO does not solve the referential game in **36 runs at 50M steps** -
including with a scripted honest sender, and with the Eccles et al.
positive-signalling bias - while a hand-coded dyad gets 4.00/4 from the same
observations. Lower action noise and a halved credit horizon do not help. So the
measurement-side results stand independent of the learner, while the
"cost/decoupling does not create signalling" claims are scoped to a standard
learner and say so.

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
