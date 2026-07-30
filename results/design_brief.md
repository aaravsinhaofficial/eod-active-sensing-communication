# DESIGN BRIEF — *From Probe to Promise: when a self-sensing pulse becomes a signal*

**Working title.** *Probe, Cue, Signal: the cost–risk–benefit conditions under which an active-sensing pulse becomes communication.*
**One-line thesis.** An EOD is first a probe (private electrolocation) and only incidentally public. We build a simulator in which the probe's private return, its public reach, its metabolic cost, and its detectability by eavesdroppers are **independently controllable**, and we map the regimes in which learning/selection converts an incidental cue into a maintained signal.

**Relation to prior work.** Singh et al. (arXiv:2511.08436 v2) built the reference MARL WEF environment but (i) defines communication purely by receiver effect, so it cannot separate *cue* from *signal* (Scott-Phillips 2008); (ii) its single causal manipulation — EOD muting — is **confounded**, because silencing a pulse destroys the emitter's own mormyromast self-image *and* its public reach simultaneously; (iii) EOD emission is free (no energetic term), and there is no predation and no evolutionary dynamics. Our Part 1 fixes the confound with a channel algebra; Part 2 supplies the cost/risk/selection axes that make the signal-vs-cue question answerable at all.

Upstream reference implementation (parity target): `/home/ec2-user/communication/wef_upstream/onpolicy/custom/fish/electric.py`, `.../electric_scene.py`, `.../sensing.py`, `.../movement.py`, `.../cfg.py`, `.../MAEFish.py`.

---

## 1. Simulator: `wef-lite` (JAX, fully vmapped)

### 1.1 State

Per environment instance: arena $W\times H$ cm (rectangular, insulating walls); $F$ agents; $N_p$ food pellets; $N_{\text{pred}}\in\{0,1\}$ predators.

Agent $i$: $\mathbf{x}_i\in\mathbb R^2$ (cm), heading $\theta_i$, size $s_i\in[0,1]$ (persistent within episode, optionally across episodes), energy reserve $E_i$, identity index $\mathrm{id}_i$, cooldowns $(\tau^{\text{eat}}_i,\tau^{\text{bite}}_i)$, GRU state $h_i\in\mathbb R^{256}$.
Pellet $n$: $\mathbf{y}_n$, orientation $\phi_n$, alive flag.
Predator: $\mathbf{x}_p,\theta_p$, hunting state $\in\{\text{search},\text{approach},\text{attack},\text{refractory}\}$.

$\Delta t = 1/83\ \mathrm{s} = 12.048$ ms (upstream `fps_sim=83`, matched to the Schuster 2001 echo latency). Episode $T=2000$ steps $\approx 24.1$ s (Part 1) / $T=4000\approx48$ s (identity & dominance assays). **Episode length is reported** — upstream never states it.

### 1.2 Kinematics (identical to upstream order-1)

$$\omega_i = u^{\text{turn}}_i\,\omega_{\max}(1+s_i),\quad \theta_i' = \mathrm{wrap}(\theta_i+\omega_i),\quad \mathbf{x}_i' = \mathbf{x}_i + \hat u(\theta_i')\,u^{\text{fwd}}_i v_{\max}(1+s_i)$$

with $v_{\max}=0.42169$ cm/step, $v_{\min}=-0.06024$, $\omega_{\max}=+0.043373$, $\omega_{\min}=-0.042169$ rad/step (upstream `cfg.py` 90th-pct linear / 95th-pct angular, already divided by 83). Collisions resolved **synchronously** (upstream resolves in agent order; this is a documented, tested divergence — see §5 R9). Eating: $\|\mathbf{x}_i-\mathbf{y}_n\|<2$ cm and $|\Delta\angle|\le\pi/8$, $\le1$ pellet/step, $\tau^{\text{eat}}\!=\!1$ decaying $1/3$ per step (motion frozen while $>0$). Biting: radius 3 cm, same cone, decay $1/5$.

### 1.3 EOD field model (faithful, closed-form, differentiable)

Every source is a point monopole or point dipole; all fields are exact Coulomb superposition with 3D kernels in a 2D world, exactly as upstream (`electric.py:57–130`). Units: positions cm → m ($\times10^{-2}$), $k=8.99\times10^9$, $\varepsilon_0=8.854\times10^{-12}$, softening $\epsilon_m=10^{-5}$ m added to the **distance before cubing**.

**(i) Active EOD.** Agent $i$ carries two monopoles at $\mathbf{x}_i \pm \tfrac{d}{2}\hat u(\theta_i)$, $d=1.0$ cm, charges $\pm q_i(t)$ with

$$q_i(t) = a_i(t)\,q_0,\qquad q_0 = 1.11\times10^{-15}\ \mathrm{C},\qquad a_i(t)\in\{0\}\cup[a_{\min},a_{\max}]$$

$a_{\min}=0.1,\ a_{\max}=3.0$ (upstream is the special case $a\in\{0,1\}$).

$$\mathbf{E}_{\text{act}}(\mathbf r)=k\sum_j \frac{q_j(\mathbf r-\mathbf r_j)}{(\|\mathbf r-\mathbf r_j\|+\epsilon_m)^3}\tag{1}$$

**(ii) Intrinsic dipoles** (EOD-independent, always on): pellet $\mathbf p_n = R(\phi_n)(0,\,1.11\times10^{-24})$ C·m; agent $\mathbf p_i = R(\theta_i)(1.11\times10^{-23},0)$; predator $\mathbf p_{\text{pred}} = R(\theta_p)(3\times10^{-23},0)$ (new; makes the predator passively sensible so hiding is not free).

$$\mathbf{E}_{\text{dip}}(\mathbf r)=k\sum_j\Big[\frac{3(\mathbf p_j\!\cdot\!\mathbf d_j)\mathbf d_j}{\|\mathbf d_j\|^5}-\frac{\mathbf p_j}{\|\mathbf d_j\|^3}\Big],\quad \mathbf d_j=\mathbf r-\mathbf r_j\tag{2}$$

**(iii) Induced dipoles** (polarizable insulating spheres, Chen et al. 2005 eq. 6):

$$\mathbf p^{\text{ind}}_n = 3\varepsilon_0 V_n\chi\,\mathbf E_{\text{act}}(\mathbf y_n),\quad V_n=\tfrac43\pi R_n^3,\ \chi=-0.5,\ R_{\text{food}}=0.25\,\mathrm{cm},\ R_{\text{fish}}=1\,\mathrm{cm}\tag{3}$$

Emitting fish get $\chi=0$ (upstream `sensing.py:622`). The driving field in (3) uses **direct EOD only** (no wall images) — a documented approximation validated in §5 R9.

**(iv) Walls.** Four first-order images per source, $x\!\to\!-x,\,2W\!-\!x$; $y\!\to\!-y,\,2H\!-\!y$; **same sign** (insulating), **reflection scale $=1.0$**. ⚠️ `cfg.ELECTRIC_CONSTANTS["reflection_scale"]=0.95` is **dead code** upstream (`electric_scene.py:379–386` never passes it); use 1.0 or parity fails. Knollen channel computed **without** reflections (upstream `sensing.py:1065`).

$$\mathbf E_{\text{tot}} = \mathbf E_{\text{act}}+\mathbf E_{\text{dip}}+\mathbf E_{\text{ind}}+\mathbf E_{\text{img}}$$

**Speed strategy.** All sources are gathered into fixed-shape padded arrays per env: $2F$ monopoles, $F{+}1$ agent/predator dipoles, $K=32$ nearest live pellets within $r_{\text{cut}}=15$ cm (upstream already caps food sensing at 15 cm), $\times 5$ (original + 4 images). With $F=4$: 220 sources × 72 sensors × 4 agents = $6.3\times10^4$ kernel evaluations per env-step, fully `vmap`-able over 2048 envs. **Throughput gate: ≥ $5\times10^4$ env-steps/s/GPU in float32 (target $1.5\times10^5$).** If the gate fails, cut $K$ to 16 and drop the $y$-images.

### 1.4 Sensing

72 sensors on the body circle $R_b=1$ cm at angles $\vartheta$, outward normals $\hat n(\vartheta)$; reading is the signed normal projection $s=\mathbf E(\mathbf r_\vartheta)\!\cdot\!\hat n(\vartheta)$ (V/m).

- **Mormyromast** $N_m=36$: 10 chin rays $\mathrm{linspace}(-\pi/6,\pi/6,10)$ + 26 body rays; sources $=\{$EOD, induced-food, induced-fish$\}$; range $[E_{M,\min},E_{M,\max}]=[5\times10^{-8},\,5\times10^{-2}]$.
- **Ampullary** $N_a=24$ uniform; sources $=\{$intrinsic$\}$ only; range $[2\times10^{-10},2\times10^{-8}]$. Degraded by conspecific EOD (noise fraction $0.05\!\to\!0.5$).
- **Knollen** $N_k=12$ **per conspecific**, identity-resolved into slot $j$ (or $j{-}1$ if $j>i$); source $=\{$that emitter's EOD$\}$ only, no reflections; threshold $E_{K,\min}=2\times10^{-7}$ (⇒ 100 cm detection).

Baseline subtraction: mormyromast subtracts the **corollary discharge** $b^{\text{CD}}$ and a **dynamic conspecific baseline** recomputed each step, $s^{\text{img}} = s^{\text{raw}}(\text{EOD}+\text{ind}) - s^{\text{raw}}(\text{EOD only})$; cons-only residuals scaled by $k_{\text{cons}}=100$. Ampullary subtracts its intrinsic baseline. Calibration constants computed **once** for a single agent at the centre of a $200\times200$ arena with EOD on — replicate exactly.

Normalization (all channels):

$$\hat s = \operatorname{sgn}(s)\cdot\frac{\log_{10}\mathrm{clip}(|s|,E_{\min},E_{\max})-\log_{10}E_{\min}}{\log_{10}E_{\max}-\log_{10}E_{\min}}\in[-1,1]\tag{4}$$

with multiplicative noise $s\leftarrow s\cdot\mathcal U(1-f,1+f)$ applied **before** clipping ($f_M=f_K=0.05$, $f_A=0.05$/$0.5$). Upstream has no noise in the paper text but does in code — **we report it and additionally sweep $f$** (§5 R14).

**New: knollen amplitude channel.** Upstream knollen is sign-only, which makes emitted **amplitude structurally unperceivable** and forecloses amplitude signalling. We add, per conspecific slot, a scalar $\ell^{i\leftarrow j}=$ normalized $\log_{10}\max_\vartheta|s|$ over $[2\times10^{-7},2\times10^{-4}]$. This conflates amplitude with distance — precisely the ambiguity that makes honesty non-trivial. Sign-only is retained as a control condition.

### 1.5 ★ The channel algebra (core methodological contribution)

Perception of EODs is routed through explicit visibility operators, applied **at the perceptual level**; physics (induced dipoles, predator detection) uses the *true* emission set unless separately gated.

| Object | Meaning |
|---|---|
| $V^{\text{self}}\in\{0,1\}^F$ | does $i$'s own pulse produce $i$'s mormyromast self-image? |
| $V^{\text{soc}}\in\{0,1\}^{F\times F}$ (off-diag) | does $j$'s pulse reach $i$'s knollen/mormyromast cons-image? |
| $V^{\text{pred}}\in\{0,1\}^F$ | is $i$'s pulse detectable by the predator? |
| $\mathcal P$ | set of *phantom* emitters (virtual sources with no body, no reward, no dynamics) |
| $\mathcal R$ | replay map: $a_{i,t}\leftarrow \tilde a_{c,t}$ from a recorded pulse train of context $c$ |
| $\Sigma_\pi$ | scramble operator on a pulse train |

The five channels of Part 1:

| Channel | Realization |
|---|---|
| (a) **private pulse** | $V^{\text{self}}_i=1$, $V^{\text{soc}}_{\cdot i}=0$ (predator gate $V^{\text{pred}}$ toggled separately) |
| (b) **social-only pulse** | $V^{\text{self}}_i=0$, $V^{\text{soc}}_{\cdot i}=1$ |
| (c) **phantom pulse** | add $p\in\mathcal P$ at prescribed $(\mathbf x_p,\theta_p,a_p,\text{times})$; $V^{\text{soc}}_{ip}=1$; no agent emitted it |
| (d) **replay pulse** | $a_{i,\cdot}\leftarrow$ recorded train from a *different* behavioural context (food-rich / food-poor / pre-bite / fleeing / predator-present), time-aligned, agent's own sensing left intact or gated per design |
| (e) **scrambled pulse** | $\Sigma_\pi$: preserve per-window pulse **count** (rate) in 250-ms windows; destroy (i) within-window timing (uniform resample), (ii) identity (permute emitter slot assignment), or (iii) both. Three sub-conditions. |

**Physical realizability note (an analytic result we will publish).** From (1)–(3), the self-image at an object at range $x$ scales as $a/x^6$ while the direct field at a conspecific at range $r$ scales as $a/r^3$. Hence

$$R_{\text{self}}(a)\propto a^{1/6},\qquad R_{\text{soc}}(a)\propto a^{1/3},\qquad R_{\text{pred}}(a)\propto a^{1/3},\qquad \frac{R_{\text{soc}}}{R_{\text{self}}}\propto a^{1/6}.\tag{5}$$

So **amplitude reduction is a cheap, physically available privacy/crypsis knob**: dropping $a$ by $30\times$ costs $1.76\times$ of private sensing range but $3.1\times$ of social and predator-detection range. This predicts amplitude-first (not rate-first) crypsis under predation — matching Reardon et al. (2011, amplitude is the sacrificial variable, $-31\%$ vs $-3\%$) and Stoddard's crypsis literature. Channels (a)/(b) are the *idealized lesions* of this continuum, and $a$ is its *realizable* version. Report both.

### 1.6 Actions (5-dim, $\mathrm{Box}(-1,1)^5$)

$u^{\text{fwd}}=\sigma(z_0)$, $u^{\text{turn}}=\tanh(z_1)$, emit $g=\mathbb 1[z_2>0]$, amplitude $a=a_{\min}(a_{\max}/a_{\min})^{(\tanh z_3+1)/2}$ (log-spaced over 1.5 decades), bite $=\mathbb 1[\sigma(z_4)>0.5]$. Refractory: at most one pulse per step (already implied by $\Delta t=12$ ms $\approx$ min IPI). Actions are **not** clipped before squashing (upstream parity).

### 1.7 Observations (113-dim for $F=4$)

$\big[\hat m^{36}\,\|\,\hat a^{24}\,\|\,\hat k^{36}\,\|\,\ell^{3}\,\|\,\hat s^{\text{cons},3}\,\|\,\tanh(z_{t-1})^{5}\,\|\,s_i\,\|\,\tilde E_i\,\|\,\tau^{\text{eat}},\tau^{\text{bite}}\,\|\,\Delta\mathbf x^{\text{ego}}_2\big]$, zero-padded for $F<4$. $\tilde E_i$ = normalized energy reserve (present only when $\kappa>0$; a matched dummy constant otherwise, to keep input dimension identical across the sweep).

### 1.8 Reward

$$\boxed{\;r^i_t=\underbrace{c_f n^{\text{food}}_{i,t}+c_p\frac{d^{\text{food}}_{i,t-1}-d^{\text{food}}_{i,t}}{P}}_{\text{foraging}}+\underbrace{c_v\mathbb 1^{\text{bitten}}_{i,t}(1{+}s_{\text{biter}}{-}s_i)+c_a\mathbb 1^{\text{bite}}+c_c\mathbb 1^{\text{coll}}+c_e\,\eta_{i,t}}_{\text{social/motor, upstream}}\;\underbrace{-\;\kappa\Big(\frac{a_{i,t}}{a_0}\Big)^{\gamma}g_{i,t}}_{\text{metabolic pulse cost}}\;\underbrace{-\;c_d\,\mathbb 1^{\text{attacked}}_{i,t}}_{\text{predation}}\;}\tag{6}$$

Coefficients (**all reported**, from `cfg.py:283–296`, then globally normalized by $\max|v|=10$ as upstream does when `NORMALIZE_REWARDS`): $c_f=10$, $c_p=1.0$, $c_v=-5$, $c_a=-0.001$, $c_c=-0.5$, $c_e=-0.1$, $P=W{+}H$. New: $c_d=-20$ (two food items), $\gamma=1.8$ (Salazar & Stoddard 2008 power-law exponent on signal power, $R^2=0.99$).

**Cost calibration (makes the axis interpretable and comparable to biology).** Run the reference policy at $\kappa=0$, measure mean per-episode food reward $\bar R_{\text{food}}$ and mean pulse count $\bar N_{\text{pulse}}$; then

$$\kappa(\tilde c)=\tilde c\,\frac{\bar R_{\text{food}}}{\bar N_{\text{pulse}}},\qquad \tilde c\in\{0,\,0.01,\,0.03,\,0.10,\,0.22,\,0.50,\,1.00\}\tag{7}$$

so $\tilde c$ reads directly as "fraction of the foraging energy budget spent on electrogenesis at the reference rate", bracketing the empirical 3% (female) – 22% (male) *B. gauderio* band.

**Interest alignment.** $r^{i,\text{eff}}_t=(1-\alpha)r^i_t+\alpha\,\frac1F\sum_j r^j_t$, $\alpha\in\{0,0.5,1\}$. Food is finite and non-replenishing (upstream `step_food_density=0`), so $\alpha=0$ is genuinely exploitative.

### 1.9 Predator (eavesdropper)

Hazard from EOD detection, using the model's own field:

$$\lambda_{i,t}=\lambda_0+\lambda_1\,g_{i,t}\,\Big[\frac{|\mathbf E^{\text{act}}_i(\mathbf x_p)|}{\theta_{\text{pred}}}\Big]^+_{\wedge 1},\qquad \Pr(\text{detect } i)=1-e^{-\lambda_{i,t}\Delta t}\tag{8}$$

$\theta_{\text{pred}}$ is set by the **range ratio** $\rho \equiv R_{\text{pred}}(a_0)/R_K$: since a dipole pair gives $|\mathbf E|\simeq 2kq_0d/r^3$, $\theta_{\text{pred}}=2kq_0 d/(\rho R_K)^3$. Sweep $\rho\in\{0(\text{no predator}),0.25,0.5,1.0,1.5\}$; $\rho\approx1.5$ is the biologically calibrated case (Hanika & Kramer: *Clarias* detects mormyrid EODs to $\sim$1.5 m vs $\sim$1 m knollen conspecific range). Baseline passive hazard $\lambda_0>0$ (predator can find a silent fish by its intrinsic dipole at short range) — **essential**, otherwise "never emit" is a free win. Predator dynamics: biased random walk; on detection, approach the detected agent at $1.3\times$ agent speed for $\le200$ steps; attack within 5 cm ⇒ $c_d$ and a 100-step incapacitation.

### 1.10 Learning

Recurrent shared-parameter **MAPPO**, CTDE, Dec-POMDP, no explicit message channel (the EOD *is* the channel). Actor & critic: 2×512 MLP (ReLU + LayerNorm) → GRU(256) → linear heads; critic sees concatenated observations. **Hyperparameters (reported, unlike upstream):** $\gamma=0.99$, GAE $\lambda=0.95$, clip $\epsilon=0.2$, lr $3\times10^{-4}$ linearly annealed, 4 PPO epochs, 4 minibatches, entropy coef 0.01 annealed to 0.001, value coef 0.5, grad-norm clip 0.5, BPTT chunk 128, 2048 parallel envs, rollout 128. Budget: $3\times10^7$ env-steps (Part 1 reference), $1\times10^7$ (phase-grid cells). Proximity shaping $c_p$ annealed to 0 over the first 30% of training (§5 R6).

**Compute.** $1680$ grid runs $\times\,10^7$ steps $=1.68\times10^{10}$ env-steps. At the $1.5\times10^5$/s/GPU target on 2 GPUs ⇒ **≈16 h**; at the $5\times10^4$/s floor ⇒ ≈4 days. Part 1 adds ~120 runs at $3\times10^7$ ⇒ +7–20 h. Evaluation/analysis: ~2000 episodes per condition, ~15% of training cost.

---

## 2. Experiments

Seeds are **independent training seeds**; every reported statistic is aggregated across all seeds (no seed selection — upstream's single-seed S1 protocol is explicitly abandoned). Evaluation is 500 episodes/seed unless noted. Statistics: cluster bootstrap over seeds (10 000 resamples), Hedges' $g$ and Cliff's $\delta$ with 95% CI, **Benjamini–Hochberg FDR $q=0.05$** across each figure's family.

| # | Experiment | Independent variables | Dependent variables | Seeds | POSITIVE result | NEGATIVE result |
|---|---|---|---|---|---|---|
| **E0a** | **Physics parity gate** (not a paper claim) | — | median relative error of the 113-d observation vs upstream NumPy over $10^4$ random scenes; trained-policy transfer | — | median err < 1%, 99th pct < 5%; upstream S1 policy in `wef-lite` reproduces food/episode within seed CI | port is wrong; fix before anything else |
| **E0b** | **Active-sensing value gate** | EOD capable vs passive-only (train from scratch) | food/episode | 8 | EOD-capable eats ≥25% more ($g>0.8$) ⇒ the probe has real private value | probe is worthless ⇒ retune ($\chi$, $E_{A}$ range, pellet size) before the paper exists |
| **E1** | **Channel decoupling, trained from scratch** (the fix for the upstream confound) | FULL / PRIVATE-ONLY ($V^{\text{soc}}=0$) / SOCIAL-ONLY ($V^{\text{self}}=0$) / NONE | food/episode, NN distance, emission rate & amplitude, patch-discovery latency, CIE, PL | 12 | The upstream muting effect **decomposes**: $\Delta_{\text{FULL}\to\text{NONE}}$ is significantly split into a self-sensing share and a social share, with social share $>0$ (FDR-corrected) | social share $\approx0$ ⇒ upstream's "communication" effect was entirely private active sensing (a strong, publishable negative) |
| **E2** | **Muting decomposition, frozen policy** (direct replication + repair of upstream) | mute-both / mute-self-only / mute-social-only / control, applied at eval | same DVs | 12 | mute-both effect $\approx$ mute-self + mute-social, with both nonzero | non-additive ⇒ report interaction; either way upstream's attribution is unsupported |
| **E3** | **Phantom pulses** | phantom amplitude $a_p\in\{0.1,0.3,1,3\}$, rate $\in\{1,3,10,20\}$ Hz, bearing $\in\{0,\pm90°,180°\}$, range $\in\{10,25,50,100\}$ cm | $\mathrm{CIE}$, behavioural PL (heading change, approach/retreat displacement at $H=41$ steps ≈ 0.5 s and $H=166$ ≈ 2 s), emission-rate response (echo) | 12 | Behavioural response exceeds the circular-shift null at $p_{\text{FDR}}<0.05$, with a **monotone or single-peaked dose–response** in rate (cf. upstream's unexplained intermediate-rate optimum) | flat response ⇒ no positive listening; the channel is inert |
| **E4** | **Context replay** (playback experiment analogue) | replayed context $c\in\{$food-rich, food-poor, pre-bite, fleeing, predator-present$\}$, matched for mean rate | receiver response vector $\mathbf y=$(Δheading toward emitter, Δspeed, Δemission rate, Δbite prob, Δdistance) at $H=166$ | 12 | Response vectors differ **by context** (permutation MANOVA over 5000 shuffles, $p<0.01$) with mean rate matched ⇒ pulses carry context-specific content, not just presence | contexts indistinguishable ⇒ the pulse signals only "a conspecific is there" |
| **E5** | **Scrambling** | $\Sigma_\pi$ variants (timing / identity / both) × {train-time, test-time} | food, CIE, PL, PS, decoding $R^2$ | 12 | timing-scramble degrades CIE/performance at matched rate ⇒ temporal code; identity-scramble degrades ⇒ identity code | only rate matters ⇒ pulse is a scalar "presence" variable |
| **E6** | ★ **Sender shaping (cue vs signal)** | receiver responsiveness at **train time**: hearing / deaf ($V^{\text{soc}}\!=\!0$ for all) / scrambled-listener; × $\tilde c\in\{0,0.03,0.22\}$ | SSI (§3.6), emission-policy KL on matched contexts, rate/amplitude/timing statistics | 12 | SSI $>$ null at high $\tilde c$ and $\approx0$ at $\tilde c=0$ ⇒ **a cost is what converts cue into signal** — the paper's headline mechanism | SSI $\approx0$ everywhere ⇒ EOD remains a cue; still a clean, important result |
| **E7** | **Cost sweep (1-D, no predator)** | $\tilde c$ (7 levels) | emission rate $\bar r$, mean amplitude $\bar a$, **which knob is turned first**, realized energy share, food | 8 | $\bar a$ and $\bar r$ both fall; realized energy share saturates in the 3–22% band; ordering of knob use is testable against Salazar & Stoddard's circadian decomposition (rate saving 62–70%, amplitude 26–72%) | no response ⇒ cost is mis-scaled; recalibrate (7) |
| **E8** | **Predation sweep** | $\rho\in\{0,0.25,0.5,1,1.5\}$, $\lambda_1$ (3 levels) | $\bar a$, $\bar r$, mortality, food, silence-before-danger latency, receiver response to conspecific **cessation** | 8 | **Amplitude falls before rate** (prediction of eq. 5), cryptic regime appears, and *silence becomes informative about danger* (nonzero $I(\text{silence};\text{predator})$) ⇒ a direct in-silico analogue of Poon & Crampton's bidirectional suppression | agents simply stop emitting entirely ⇒ $\lambda_0$ too low, retune |
| **E9** | ★ **Full phase grid** | $\tilde c$ (7) × $\rho$ (5) × $\alpha$ (3) × identity {persistent, reshuffled} (2) = 210 cells | full metric vector $\mathbf f$ (§3.7), regime label | 8/cell (1680 runs) | ≥4 of the 6 target regimes appear as contiguous, seed-stable regions; rule-based and unsupervised labels agree (ARI $>0.6$) | regimes are scattered/seed-unstable ⇒ report as a continuum, not phases (still a result, but reframe) |
| **E10** | **Identity, dominance, honesty** | persistent vs reshuffled partners; fixed vs random re-pairing; size dispersion $\sigma_s\in\{0.05,0.25\}$; $\tilde c$ (3) | rank–rate relation, submissive-silence index, amplitude–size regression $R^2$ (index-likeness, cf. Gavassa's 96%), deception index DEC, win ratio, Theil index | 8 | Subordinates reduce rate/amplitude near dominants (submissive silence); amplitude–size $R^2$ **increases with $\tilde c$** (costly ⇒ honest, the Gavassa/Sir-Philip-Sidney prediction); DEC $>$ null only at low $\tilde c$ & $\alpha=0$ | no rank structure ⇒ episodes too short; extend to $T=4000$ or multi-episode lifetimes |
| **E11** | **Control battery** (runs alongside E1–E10) | separate emission-head network; scrambled-training; retrained no-channel baseline; patch-confined random-walker bot resident; placebo-cost control | PS, SC, CIE, food | 8 each | PS collapses under separate-head/scrambled-training ⇒ the original PS was **not** a shared-trunk artifact (Lowe et al.'s failure mode is excluded) | PS survives scrambling ⇒ PS was an artifact; all signalling claims must be withdrawn |
| **E12** | *(optional, 1-D slices only)* **Evolutionary outer loop** | population 64 genomes $\theta_g=(\log a_0, \text{emission bias}, \text{knollen gain})$, truncation selection top-25%, Gaussian mutation $\sigma=0.1$, 200 generations, at 5 $(\tilde c,\rho)$ points | genome trajectories, ESS emission rate | 5 replicate populations/point | Evolved emission rates match the RL-learned rates within CI ⇒ "learning as adaptation" proxy validated; and evolution satisfies the Scott-Phillips "evolved because of that effect" clause directly | divergence ⇒ report the RL result as *within-lifetime adaptation only* and soften all "evolve" language |

---

## 3. Metrics (exact definitions)

Throughout: $\pi_i(\cdot\mid h^i_t,o^i_t)$ is the receiver's policy; $m^j_{t-W:t}$ is emitter $j$'s pulse train (amplitudes and times) over a window $W=41$ steps ($\approx0.5$ s).

### 3.1 Causal influence of the EOD (CIE) — multi-step, GRU-consistent

Single-step influence (Jaques et al. Eq. 1, with their Eq. 3 conditioning on the receiver's recurrent state to block back-doors) is insufficient here because knollen effects are temporally extended. We use the Eccles-style multi-step form:

$$\mathrm{CIE}^{j\to i}_t(W)\;=\;D_{\mathrm{KL}}\!\Big[\pi_i\big(\cdot\mid h^i_t,\,o^i_t\big)\;\Big\|\;\pi_i\big(\cdot\mid \tilde h^{i,\neg j}_t,\,\tilde o^{i}_t\big)\Big]\tag{9}$$

where $\tilde h^{i,\neg j}_t$ is obtained by **re-rolling the receiver's GRU from $t-W$** under the intervention $\mathrm{do}\big(m^j_{t-W:t}=\tilde m\big)$, holding every other input on its factual path. Three null distributions for $\tilde m$, all reported:

1. **Time-shift null (primary)**: $\tilde m$ = $j$'s own train from a random circular shift $\ge2$ s, preserving marginal rate/amplitude/burstiness and destroying temporal contingency.
2. **Marginal resample**: $\tilde m\sim \hat p(m)$ pooled across the condition.
3. **Zero** (reported only for comparability with upstream; Lowe et al. explicitly warn this is OOD).

Group-level: $\mathrm{CIE}=\mathbb E_{t,i,j}[\mathrm{CIE}^{j\to i}_t]$, significance vs a within-condition shift-null with $n=200$ shifts, threshold = 97.5th percentile.

**Behavioural CIE** (policy-independent, the one a biologist will trust):

$$\mathrm{CIE}^{\text{beh}}(H)=\big\|\;\mathbb E[\mathbf y_{t+H}\mid \mathrm{do}(m)] - \mathbb E[\mathbf y_{t+H}\mid \mathrm{do}(\tilde m)]\;\big\|_2,\quad \mathbf y=(\Delta\theta,\Delta\|\mathbf x\|,\Delta r^{\text{emit}},\Pr[\text{bite}])$$

reported in physical units (deg, cm, Hz, prob) at $H\in\{41,166\}$.

### 3.2 Positive signaling (Lowe Def. 3.1, quantified)

Let $M^j$ be a window descriptor $\big(\text{count},\ \overline{\log a},\ \mathrm{CV}_{\text{IPI}},\ \text{burst flag}\big)$ and $S^j$ the sender's private state variable of interest. Then

$$\mathrm{PS}(S)=\frac{I(M^j;S^j)}{H(S^j)}\in[0,1],\tag{10}$$

estimated by KSG for continuous $S$ and by plug-in with Miller–Madow correction for discrete $S$; significance by 1000 permutations of $S$ within matched geometry strata (distance-to-nearest-conspecific decile) to prevent geometry from manufacturing dependence.

**Mandatory artifact controls** (Lowe et al. showed PS arises from shared trunks alone): report PS additionally for (i) a policy with a **separate emission network**, (ii) a policy trained with **scrambled received messages**, (iii) an **untrained emission head**. PS is only reported as evidence of signaling if it collapses in (i)–(iii).

### 3.3 Positive listening (Lowe Def. 3.2, quantified)

$$\mathrm{PL}_i \;=\; \mathbb E_t\Big[\big\|\pi_i(\cdot\mid h^i_t,o^i_t)-\pi_i(\cdot\mid \tilde h^{i,\neg j}_t,\tilde o^i_t)\big\|_1\Big]\tag{11}$$

(the L1 form of Eccles et al., empirically more stable than KL), with the same time-shift null. Plus the **behavioural** version $\mathrm{PL}^{\text{beh}}=\mathrm{CIE}^{\text{beh}}$ above, and the cheap necessary-condition check $\mathrm{MIN}=\|W^1_{\text{knollen}}\|_F$ (zero norm ⇒ no listening).

### 3.4 Information content about food / danger / dominance / movement

Decode **from the receiver-observable channel only** (knollen sign pattern + amplitude scalar + emitter bearing history over $W$), never from ground-truth fields:

$$\mathcal I(Y)=1-\frac{\mathbb E\big[(Y-\hat g_\psi(m^{j}_{t-W:t}))^2\big]}{\mathrm{Var}(Y)}\quad\text{(continuous)},\qquad \mathrm{AUC}\ \text{(binary)}\tag{12}$$

with **grouped cross-validation over held-out episodes** and a time-shift baseline subtracted. Targets:

| Target | Definition |
|---|---|
| **Food** $Y_{\text{food}}$ | # live pellets within 10 cm of emitter at $t$; and binary "emitter ate within $[t,t+41]$" |
| **Danger** $Y_{\text{dang}}$ | binary predator within $\rho R_K$ of emitter at $t$; also decoded from **silence runs** (max inter-pulse gap in window) |
| **Dominance** $Y_{\text{dom}}$ | emitter size $s_j$; and realized win ratio $n^{\text{given}}/(n^{\text{given}}+n^{\text{received}})$ |
| **Intended movement** $Y_{\text{mov}}$ | emitter's heading change and displacement over $[t,t+41]$, decoded via paired cos/sin models |

Decoders: ridge (α selected by inner CV) and a 2-layer MLP; report the max, plus a linear-probe/MLP gap as a nonlinearity diagnostic.

### 3.5 Honesty and deception

Following Skyrms (2010), the informational content of a realized signal $m$ about state $y$ is the vector component

$$\iota_y(m)=\log\frac{\hat p(y\mid m,\,z)}{\hat p(y\mid z)},\qquad z=\text{receiver-observable context (distance, bearing, own state)}.\tag{13}$$

**Honesty index** $\mathrm{HON}(Y)=\mathcal I(Y)$ from (12) (calibration-checked: reliability diagram slope of $\hat p(y|m)$ vs empirical frequency; slope 1 = calibrated/honest).

**Misinformation**: signal $m_t$ is misinformative about the realized state $y^\ast_t$ iff $\iota_{y^\ast_t}(m_t)<0$.

**Deception index** (all three Maynard Smith–Harper clauses, evaluated counterfactually):

$$\mathrm{DEC}=\frac{\Pr\Big(\iota_{y^\ast_t}(m_t)<0\ \wedge\ \Delta u^{\text{send}}_{t:t+H}>0\ \wedge\ \Delta u^{\text{recv}}_{t:t+H}<0\Big)}{\Pr\big(\iota_{y^\ast_t}(m_t)<0\big)}\;-\;\mathrm{DEC}_{\text{null}}\tag{14}$$

where $\Delta u^{\text{send}},\Delta u^{\text{recv}}$ are cumulative reward differences against the **matched time-shift counterfactual** (what the pair would have obtained had the signal been non-contingent), and $\mathrm{DEC}_{\text{null}}$ is the same quantity under the shift-null. $\mathrm{DEC}>0$ (FDR-corrected) ⇒ systematic deception.

### 3.6 ★ Sender Shaping Index (SSI) — the cue/signal discriminator

Scott-Phillips: a **signal** requires that the sender's production evolved *because of* its effect on receivers; a **cue** does not. Operationalization: train two populations from scratch, identical except that receivers are hearing ($\mathcal H$) or deaf ($\mathcal D$; $V^{\text{soc}}\equiv0$ throughout training). Compare emission policies on a **common, matched context set** $\mathcal C$ (states drawn from a third, neutral rollout distribution, stratified by distance-to-conspecific, food proximity, predator presence):

$$\mathrm{SSI}=\frac{\mathbb E_{c\in\mathcal C}\;D_{\mathrm{JS}}\big[\pi^{\mathcal H}_{\text{emit}}(\cdot\mid c)\,\big\|\,\pi^{\mathcal D}_{\text{emit}}(\cdot\mid c)\big]}{\mathbb E_{c\in\mathcal C}\;D_{\mathrm{JS}}\big[\pi^{\mathcal H,\text{seed }1}_{\text{emit}}\,\big\|\,\pi^{\mathcal H,\text{seed }2}_{\text{emit}}\big]}\;-\;1\tag{15}$$

i.e. **between-condition divergence normalized by within-condition seed divergence**. $\mathrm{SSI}\le0$ ⇒ emission is unshaped by receiver responsiveness ⇒ **cue**. $\mathrm{SSI}>0$ with FDR-corrected significance ⇒ **signal**. A secondary, stronger variant uses the E12 evolutionary loop (does the *heritable* emission trait differ between hearing and deaf worlds?).

**Net-benefit test** (Noukhovitch et al.: high influence in competitive settings can be *manipulation* or *cue-reading*, not communication): a condition qualifies as **communication** only if additionally $\Delta u^{\text{send}}>0$ **and** $\Delta u^{\text{recv}}>0$ against the shift-null, both FDR-corrected.

### 3.7 Regime feature vector and classifier

Per grid cell, aggregate across seeds:

$$\mathbf f=\big[\ \bar r,\ \bar a,\ \mathrm{CIE},\ \mathrm{PS},\ \mathrm{PL},\ \mathrm{SSI},\ \mathrm{HON}(s_j),\ \mathrm{DEC},\ \Delta u^{\text{send}},\ \Delta u^{\text{recv}},\ \mathrm{SIL},\ \mathrm{MORT}\ \big]\in\mathbb R^{12}$$

with $\mathrm{SIL}=1-\bar r_{\text{low-rank}}/\bar r_{\text{high-rank}}$ evaluated within 5 cm of a dominant, and $\mathrm{MORT}$ = predation mortality per episode. All thresholds $\varepsilon_\bullet$ are the 97.5th percentile of the corresponding shift-null pooled across the grid — **not hand-set**.

---

## 4. Phase diagram

**Primary plane (per panel):** $x=\log_{10}\tilde c$ (metabolic cost as fraction of foraging budget, 7 levels, with the empirical 3–22% band shaded); $y=\rho=R_{\text{pred}}/R_K$ (eavesdropper range relative to conspecific range, 5 levels, with $\rho\approx1.5$ marked as the *Clarias*/mormyrid calibration).
**Panel rows:** $\alpha\in\{0,0.5,1\}$ (competition → cooperation).
**Panel columns:** identity persistent vs reshuffled (reputation on/off).
**Secondary sweeps (1-D slices, reported in supplement):** $R_K/L$ (receiver range vs arena size, controlling encounter rate), group size $F\in\{2,4,8\}$ at fixed density, knollen amplitude channel on/off, sensor noise $f$.

**Regime rules** (deterministic, auditable; evaluated in this order):

| Regime | Rule |
|---|---|
| **P — private active sensing** | $\bar r>\varepsilon_r$ ∧ $\mathrm{CIE}\le\varepsilon_C$ ∧ $\mathrm{SSI}\le0$ |
| **U — honest public information (cue / eavesdropping)** | $\mathrm{CIE}>\varepsilon_C$ ∧ $\mathrm{PL}>\varepsilon_L$ ∧ $\mathrm{SSI}\le0$ ∧ $\Delta u^{\text{recv}}>0$ |
| **C — cooperative communication (signal)** | $\mathrm{CIE}>\varepsilon_C$ ∧ $\mathrm{SSI}>\varepsilon_S$ ∧ $\Delta u^{\text{send}}>0$ ∧ $\Delta u^{\text{recv}}>0$ ∧ $\mathrm{DEC}\le\varepsilon_D$ |
| **D — deceptive / competitive signalling** | $\mathrm{SSI}>\varepsilon_S$ ∧ $\big(\mathrm{DEC}>\varepsilon_D$ ∨ $(\Delta u^{\text{send}}>0\wedge\Delta u^{\text{recv}}<0)\big)$ |
| **S — submissive silence** | $\mathrm{SIL}>0.5$ (FDR-sig.) ∧ silence conditional on dominant proximity ∧ $\mathrm{CIE}$(silence)$>\varepsilon_C$ |
| **K — cryptic low-rate sensing** | $\bar a<0.5\,\bar a_{\rho=0}$ ∧ $\mathrm{MORT}$ significantly below the loud-policy counterfactual ∧ $\mathrm{CIE}\le\varepsilon_C$ |

Cells may satisfy multiple rules (e.g. K∧S); report the full 6-bit membership vector and colour by the dominant rule, with hatching for co-membership. **Validation:** fit a 6-component GMM on standardized $\mathbf f$ (seeds as replicates); report ARI between unsupervised clusters and rule labels (target > 0.6), plus a threshold-sensitivity map over $\varepsilon_\bullet\times[0.5,2]$.

**Pre-registered theoretical boundary.** From the Sir Philip Sidney game with relatedness $k$, the minimum cost of a believable signal is $c^\ast=b-kd$; with unrelated partners ($k=0$), $c^\ast=b$ — honest costly signalling of quality requires cost at least the benefit of faking. We will overlay the predicted $c^\ast$ line (with $b$ estimated from the measured payoff of a successful size-bluff) on the $\tilde c$ axis and test whether the **U→C/D** boundary coincides with it. We additionally pre-register Huttegger & Zollman's result that pooling equilibria have the larger basin of attraction, predicting **P/U to dominate the low-$\tilde c$ half of the diagram** even where C is feasible.

---

## 5. Risks and the controls that kill them

| # | Risk (what makes the result trivial or confounded) | Control |
|---|---|---|
| **R1** | **The probe has no private value** — if passive ampullary sensing already finds food (upstream ranges: ampullary 4.6 cm vs mormyromast 5 cm for prey), the whole cost/benefit story is vacuous. | **E0b gate**: require ≥25% food advantage for EOD-capable agents; if it fails, retune $\chi$/pellet radius/ampullary range *before* the sweep, and report the tuning. |
| **R2** | **"Social-only pulse" is physically impossible**, inviting the charge that Part 1 is a cartoon. | Frame channels as **perceptual lesions** (the electrosensory analogue of optogenetic silencing), state it explicitly; *and* pair every idealized lesion with the **physically realizable amplitude continuum** of eq. (5), which produces the same privacy/publicity trade-off with real physics. |
| **R3** | **Test-time ablation shock** — zeroing a log-normalized channel puts it at a value never seen in training; the deficit measures OOD brittleness, not channel function (Lowe et al.'s explicit warning; this is exactly what upstream's ablations do). | **Train-from-scratch is primary** (E1, E5, E6). Test-time ablation (E2) is reported only as a replication of upstream, side by side. Time-shift nulls (not zeros) are the primary counterfactual everywhere. |
| **R4** | **Positive signaling is a shared-trunk artifact** — Lowe et al. showed SC survives message scrambling and appears with an untrained comm head; MI(message; state) proves nothing. | **E11 battery**: separate emission network, scrambled-training, untrained-head. PS is reported as evidence only if it collapses in all three. Never report MI alone as evidence of communication. |
| **R5** | **Manipulation mistaken for communication** — high CIE is compatible with the receiver reading the sender against the sender's interest (Noukhovitch et al.), especially at $\alpha=0$. | Mandatory **two-sided net-benefit test** ($\Delta u^{\text{send}}>0$ ∧ $\Delta u^{\text{recv}}>0$) in the C rule; cells failing it are labelled U or D, never C. |
| **R6** | **Reward-shaping leakage** — $c_p$ proximity shaping makes "food nearby" trivially entangled with movement, inflating $\mathcal I(Y_{\text{food}})$. | Anneal $c_p\to0$ by 30% of training; run a no-shaping arm at 3 grid points; report $\mathcal I(Y_{\text{food}})$ both ways. |
| **R7** | **Seed cherry-picking** (upstream selected S1 of 20 using an ethological criterion that included the very size–food relation it then reported). | No seed selection at any stage; 12/8 seeds fully reported; seed is a random effect in every model; per-seed scatter shown in every figure. |
| **R8** | **Phases are threshold artifacts.** | Null-calibrated thresholds; sensitivity map over $0.5$–$2\times$ each $\varepsilon$; unsupervised GMM cross-check (ARI); seed-stability map (fraction of seeds per cell agreeing with the modal label). |
| **R9** | **Port infidelity** — dead `reflection_scale=0.95`, first-order-only images, $K$-nearest food truncation, no image-driven induction, synchronous collisions, RNG ordering. | E0a parity gate (<1% median obs error, policy transfer); ablation of each approximation ($K\in\{8,16,32,64\}$, images on/off, induction driver with/without images) with error budget reported in supplement. |
| **R10** | **Cost confound** — agents may reduce EODs simply because *any* action cost suppresses activity. | **Placebo-cost control**: identical $\kappa$ applied to the (behaviourally irrelevant) bite logit magnitude at 3 grid points; movement cost held fixed across the entire sweep; report $\bar r$ vs $\bar v$ (swim speed) to show the response is emission-specific. |
| **R11** | **Predation trivially selects total silence.** | $\lambda_0>0$ (passive intrinsic-dipole hazard) guarantees an interior optimum; verify analytically first with a 1-D toy model $\max_a\ [\text{food}(R_{\text{self}}(a))-\kappa a^{1.8}-c_d\lambda_1 R_{\text{pred}}(a)^2]$ and pre-register the predicted $a^\ast(\tilde c,\rho)$ contour. Reject a cell as degenerate if $\bar r<0.01$ Hz. |
| **R12** | **Density/group-size confounds** with cost and range. | Arena area scaled with $F$ to hold agent and food density fixed; $R_K/L$ reported for every cell. |
| **R13** | **Reputation/dominance cannot emerge in a 24-s episode.** | $T=4000$ (48 s) plus **multi-episode lifetimes** (5 consecutive episodes, persistent size/identity/energy) for E10; explicit fixed-partner vs random-repairing contrast to test whether repeated interaction is required for honesty. |
| **R14** | **Noiseless physics** makes "detection range" pure geometry and removes SNR, which is the heart of electrosensory ecology (upstream's paper reports no noise at all). | Multiplicative receptor noise is on by default ($f=0.05$) and swept $f\in\{0,0.05,0.2\}$ at 3 grid points; all detection claims are stated as signal-detection statements (d′) not geometric ranges. |
| **R15** | **Multiple comparisons** (upstream applies none across a large panel family). | BH-FDR $q=0.05$ within each figure family; all effect sizes with CIs; a pre-registration document frozen before the grid runs. |

---

## 6. Figure list

1. **Model & channel algebra.** (a) Arena schematic with field lines and the three receptor populations. (b) The visibility operators $V^{\text{self}},V^{\text{soc}},V^{\text{pred}}$ as matrices, with the five channels as instantiations. (c) Analytic scaling of eq. (5): $R_{\text{self}}\propto a^{1/6}$, $R_{\text{soc}},R_{\text{pred}}\propto a^{1/3}$, with simulated points overlaid. (d) Parity gate: observation-vector error vs upstream; throughput (env-steps/s) vs $K$.
2. **Decoupling the dual function.** (a) E1 trained-from-scratch food/NN-distance/emission-rate for FULL / PRIVATE-ONLY / SOCIAL-ONLY / NONE. (b) E2 muting decomposition showing that the upstream "communication" effect splits into self-sensing and social components. (c) Waterfall attributing the total muting effect to each component with CIs.
3. **Phantom, replay, scramble.** (a) Phantom dose–response surfaces (rate × amplitude × bearing) for behavioural CIE, with shift-null band. (b) Context-replay response vectors (radar/PCA) per replayed context with permutation-test significance. (c) Scrambling ladder: intact → timing-scrambled → identity-scrambled → both, on CIE and food.
4. **What a pulse carries.** Decoding $\mathcal I$/AUC for food, danger, dominance, intended movement, from receiver-observable channels only, vs shift-null; ×(receiver range, cost, predation). Inset: danger decoded from **silence** rather than pulses.
5. **Cue → signal.** SSI (eq. 15) as a function of $\tilde c$ and receiver responsiveness (hearing/deaf/scrambled), with the Scott-Phillips 2×2 (signal/cue/coercion) as the interpretive frame; the artifact-control panel (separate head, scrambled training, untrained head) showing PS collapse.
6. **The price of a pulse.** E7: emission rate and amplitude vs $\tilde c$; realized energy share with the 3–22% empirical band; which knob is turned first, compared to the measured circadian rate/amplitude decomposition of *B. gauderio*.
7. **Eavesdroppers.** E8: amplitude-before-rate crypsis vs $\rho$; mortality vs emission policy; silence-before-danger latency and its receiver-side consequences (in-silico analogue of bidirectional signal suppression).
8. ★ **The phase diagram.** $\log_{10}\tilde c$ × $\rho$, panelled by $\alpha$ and identity persistence, coloured by regime, hatched for co-membership, with the pre-registered Sir Philip Sidney $c^\ast=b-kd$ boundary overlaid; side panels: seed-stability map, threshold-sensitivity map, rule-vs-GMM ARI.
9. **Honesty, dominance, deception.** Amplitude–size regression $R^2$ vs $\tilde c$ (index-likeness; compare Gavassa's 96%); submissive-silence index vs rank and cost; DEC vs $\alpha$ and identity persistence; win-ratio / Theil inequality by regime.

---

### Deliverables and repository layout

`wef-lite/` — `physics.py` (eqs. 1–4, JAX), `channels.py` (visibility algebra), `env.py` (state/dynamics/reward eq. 6), `predator.py` (eq. 8), `train_mappo.py`, `metrics/{cie.py,ps_pl.py,decode.py,deception.py,ssi.py,regime.py}`, `parity/test_vs_upstream.py`, `sweeps/phase_grid.yaml`, `PREREGISTRATION.md` (frozen before E9). Every reward coefficient, hyperparameter, episode length, and threshold is emitted to a per-run `config.json` and reported in the paper — the three most-cited reproducibility gaps in the prior work.