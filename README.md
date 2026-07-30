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
them. This repository makes the three independently switchable.

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

# 2. train every condition (168 runs; ~1.5 h on 2 GPUs)
python3 scripts/launch_all.py

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
