# Transfer Utility Is Not a Property of the Task Pair: A Systematic Failure of Proxy Prediction and the Cost of Direct Measurement

<!--
2026-08-07 标题整改。原标题为
"An Impossibility Characterization and the Minimum Measurement That Suffices"，
违反 PAPER_CLAIMS_20260804.md §5 的两条禁写：

  第 8 条  不得使用一般意义的 "impossibility" 措辞（标题、摘要、正文皆然），
           除非给出形式证明。正确表述是 "empirical systematic failure of
           tested proxy families"。
  第 9 条  不得称 K*=10000 是 minimum sufficient measurement，
           只能称"已测试预算 {2k,5k,10k} 中最小的稳健 horizon"。

R0 已在 PAPER_CLAIMS 中降级，但正文与大纲当时未同步整改。
-->


> Draft v1, 2026-08-06. Every number in this draft is traceable to a frozen
> adjudication output listed in Appendix A. Claim-by-claim provenance and scope
> live in `docs/PAPER_CLAIMS_20260804.md`; section/figure planning in
> `docs/PAPER_OUTLINE_20260804.md`.
>
> **Status of each contribution is marked inline.** Nothing here may be
> strengthened beyond what the cited adjudication supports.

---

## Abstract

Cross-task transfer in reinforcement learning requires two decisions: *whether*
to use a source policy at all, and *which* one. The dominant approach is to
predict transfer utility from a cheap, pre-injection quantity. We report a
systematic refutation of that approach on HumanoidBench: **twelve signal
families spanning seven signal spaces**, each independently pre-registered and
independently adjudicated, all fail. Beyond the empirical count we give **two
principled counterexamples** showing that this class of quantities cannot carry
the decision even in principle: (i) two targets that share a *byte-identical*
reward implementation yield utilities of `+56.95` and `+0.19 [−5.35, +5.72]`,
so any predictor that reads **only the reward specification** receives identical
inputs and must produce identical outputs (the two targets do differ in terrain
geometry, transition dynamics, initial state distribution and MJCF, so this
bounds reward-only predictors, not every static-specification predictor);
(ii) for the weakest possible use — one-sided
exclusion rather than ranking — the admissible threshold interval is
`14.302 < θ < 1.814`, the **empty set**, independently of how the threshold is
chosen. What remains is direct measurement. We show that a short racing
procedure recovers the correct source with `K* = 10000` learner steps per arm
(3/3 seeds in each of two independent batches), that a *single* such
measurement simultaneously decides admission (`false_admit = 0`,
`false_reject = 0` over nine target–seed cells), and that choosing correctly is
worth `4.95×` in steps-to-threshold. Assembled into an automatic decision chain,
the system makes 9/9 decisions consistent with known ground truth and beats both
a scratch baseline and a zero-cost-signal baseline under step-aligned accounting
(3/3 each). We report the unfavourable accounting as well: once the racing cost
is charged in full, the advantage survives on only one of three targets.
Accordingly we claim **higher and more stable final performance**, not improved
sample efficiency.

---

## 1. Introduction

Given a set of frozen source policies and a target task, a transfer agent must
decide whether any source is worth using and, if so, which. The standard move is
to design a cheap statistic `X` — behavioural return, immediate reward, gradient
alignment, critic advantage, reward-function structure, task metadata — and to
assume `X → U`, where `U` is the delayed effect of injecting that source on the
learner's eventual performance.

This paper argues that the move fails, and that it fails for a structural reason
rather than for want of a better statistic.

Our argument proceeds in six steps, each written to block a specific objection:

1. We formalise the decision as a *causal intervention label* `U` measured
   source-free (Section 2).
2. The natural approach is prediction: estimate `U` from a cheap `X`.
3. We report twelve independently adjudicated families that all fail
   (Section 3.1). *Objection: you did not try the right statistic.* We answer
   with a coverage table over seven distinct signal spaces.
4. **An empirical count is not an impossibility.** No number of failures rules
   out an unseen family; this objection must be answered structurally, not by
   adding a thirteenth family.
5. We give two principled counterexamples (Section 3.2). The first exhibits two
   targets whose **reward specifications are byte-identical** yet whose utilities
   differ, so no function of the reward specification alone can separate them.
   The second shows the feasible threshold interval is empty — a conclusion
   invariant to the choice of threshold rule.
6. Measurement is therefore the route we take. We quantify its cost at the
   budgets we tested and show one measurement decides both questions
   (Sections 4–5).

**Contributions.**

- **C1** An empirical systematic failure of zero-cost transferability
  prediction: twelve signal families over seven spaces, plus two principled
  counterexamples (Section 3). We do **not** claim an impossibility theorem —
  no formal proof is given, and no number of failed families constitutes one.
- **C2** A measurement horizon that suffices at the budgets we tested:
  `K* = 10000` is the smallest robust horizon among `{2k, 5k, 10k}`, with a
  single measurement deciding both admission and selection, and a quantified
  cost–benefit (Section 4). Whether a smaller untested budget would also
  suffice is open.
- **C3** An automatic end-to-end decision chain whose 9/9 decisions match known
  ground truth and which outperforms two baselines under step-aligned
  accounting — with the unfavourable accounting reported in full (Section 5).

We deliberately do **not** claim a new transferability metric. C1 is precisely
the statement that such a metric is unavailable in the spaces we examined; the
racing procedure of C2 is a *measurement*, not a predictor.

---

## 2. Problem Setup

**Environment.** HumanoidBench with the `h1hand` morphology (76 DoF, 61
position-controlled actuators, two Shadow hands). Observations are the
concatenation of full `qpos` and `qvel`; for locomotion targets this is 151
dimensions. Control runs at 50 Hz with 1000-step episodes. The learner is
FastTD3 (distributional double critic with clipped double-Q), 128 parallel
environments.

**Sources.** Three frozen locomotion policies — `stand`, `walk`, `run` — each
trained with the same learner on its own task.

**The intervention label.** For source `i`, learner state `θ_t`, dose `d`, and
horizon `K`:

```
U_i(t, d, K)  =  J_sf( θ_i at t+K )  −  J_sf( θ_student at t+K )
```

where `J_sf` is a **source-free** evaluation: the source is absent at evaluation
time, and only the student policy is measured. This is the invariant convention
throughout: *what is finally evaluated is always the student without the source.*
Evaluation uses a frozen panel of 16 eval seeds × 8 ranks = 128 deterministic
episodes, with reset seed `eval_seed × 1000 + rank`, identical bit-for-bit
across arms. Paired comparisons are taken episode-by-episode on this panel.

**Injection channel.** Sources are injected through a behaviour-bootstrap
channel: at each latch expiry an arm is drawn from a single categorical over
admitted sources plus the student, and the drawn actor runs for a 25-step
call-and-return segment. Source transitions additionally receive a replay quota.
Unless stated otherwise the source mass is 0.5, verified per checkpoint.

---

## 3. Pillar I: Zero-Cost Prediction Is Not Available

### 3.1 Twelve signal families across seven spaces

Each family below was pre-registered and adjudicated independently, with the
decision rule committed to version control before the corresponding data
existed.

| # | Family | What it measures | How it failed |
|---|---|---|---|
| 1 | zero-shot behaviour return / displacement | source's immediate performance on target | sign inverts (below) |
| 2 | `T⁰` | immediate effect at `t=0` | sign disagrees with delayed utility |
| 3–6 | SIV / SHU / adaptive revocation / lease | immediate segment reward | guiding sources do the "dirty work" and score low |
| 7 | update-space influence | gradient inner product | fails **and inverts ranking** |
| 8 | `T^critic` | `E[min Q(s,π_i) − min Q(s,π_stu)]` | sign biased negative by construction |
| 9 | bottleneck-aligned coverage | measured reward-component coverage | no increment over per-step reward |
| 10 | per-state QMP fidelity | `argmax_i min_h Q_h(s,π_i(s))`, **no aggregation** | degenerates to student; critic advantage negative in 18/18 cells |
| 11 | specification matching | reward constants + source training constants | refuted on its own best case (§3.2) |
| 12 | task progress (zero-training) | zero-shot forward displacement | inverts by `7.9×`; empty threshold interval (§3.2) |

Family 10 is worth isolating: it removes aggregation entirely and asks the
target's own critic, state by state. The immediate critic advantage of the source
over the student is negative in **18/18** `(task, source, seed)` cells —
including the case where the true utility is `+56.95`. This rules out
"the aggregation was wrong" as an explanation: the defect is in the quantity
being aggregated.

### 3.2 Two principled counterexamples

**Counterexample A — byte-identical reward specification, different utilities.**
The HumanoidBench targets `slide` and `stair` are both implemented by the same
class, `ClimbingUpwards`, and share a **byte-identical** `get_reward`, including
all numeric constants. The measured utility of the `walk` source differs:

```
slide :  U(walk) = +56.95            (argmax across sources; stable over 6 learners)
stair :  U(walk) = +0.19  [−5.35, +5.72]   (n = 3; crosses zero)
```

Any quantity that reads only the **reward specification** observes
**identical inputs** on these two targets and therefore cannot produce this
difference — regardless of how it is designed.

**Scope of this counterexample (do not overstate).** `slide` and `stair` differ
in terrain geometry, transition dynamics, initial state distribution, and MJCF.
The counterexample therefore refutes *reward-only* predictors; it does **not**
refute a predictor that reads the full static task specification. An earlier
draft claimed the latter and has been corrected.

Note also that the second value is
statistically indistinguishable from zero, so the contrast is not "300× smaller"
but "indistinguishable from no effect versus a stable positive effect."

**Counterexample B — the feasible threshold interval is empty.**
Family 12 weakens the requirement as far as possible: no ranking, only *one-sided
exclusion* (reject a source whose zero-shot task progress is near zero), and it
replaces return with task progress, which is immune to the reward-composition
confounds that sink family 1. Measured zero-shot forward displacement (metres,
32 episodes, deterministic):

| target | stand | walk | run | true best source |
|---|---:|---:|---:|---|
| crawl (all sources harmful) | 0.221 | 3.664 | **14.302** | none (`U` = −448 / −217 / −208) |
| hurdle | 0.188 | 8.717 | **22.521** | `run` (`U` = +379.66) |
| slide | 0.183 | **1.814** | 1.753 | `walk` (`U` = +56.95) |

For the screen to work it must simultaneously satisfy

```
P(run, crawl) < θ        (reject a harmful source)
P(walk, slide) > θ        (retain a useful source)
i.e.   14.302 < θ < 1.814        →  ∅
```

The harmful source travels **7.9× farther** than the useful one. No threshold
exists; the conclusion is invariant to the threshold rule, to normalisation, and
to how the displacement arises mechanically.

A related intuition also fails: `walk` on `slide` is blocked by the incline to
94% of its flat-ground displacement (31.388 m → 1.814 m) and is nonetheless the
only stably positive source. **"Structurally blocked on the target" does not
imply "useless for learning."**

### 3.3 Why: `U` is not a point function of `(source, target)`

Three independent lines converge:

- **Direction asymmetry.** Similarity is symmetric; utility is not. In a
  pre-registered sibling test, `slide → stair` gives `+15.40 [+5.72, +25.08]`
  (3/3 positive) while the reverse gives `−20.79 [−31.61, −9.97]` (0/3).
  *The negative direction is the robust half*: the sibling arm there held a
  2.5–3.3 pp dose advantage and still lost, so dose cannot explain it. The
  positive direction must be discounted, since its dose advantage points the
  same way as its win.
- **Channel attribution flips across seeds.** Decomposing the joint effect into
  behaviour and replay channels on `door` yields opposite attributions on
  different learner seeds, with per-episode `|U|/SE` of 10–20 — not noise.
- **The sign itself is not learner-invariant.** On `door`, 18/18 per-seed effects
  were negative across two batches; a third batch produced 2/9 positive,
  including `+36.32 ± 3.95`.

The correct object is therefore a conditional distribution

```
U ~ p( U | source, target, θ_t, D_t, occupancy_t, channel, dose, K )
```

whereas all twelve families estimate a point function of `(source, target)`, at
most with `t`.

**Scope.** This is not an exhaustive proof. Twelve families cover seven spaces
(behaviour, immediate reward, gradient, critic, reward structure, task
definition/static specification, task progress); an unexamined family is not
excluded. The claim is: *within the spaces examined, no quantity depending only
on `(source, target)` predicts delayed learning utility* — and, by §3.2, two of
those failures are structural rather than empirical.

---

## 4. Pillar II: Direct Measurement and What It Costs

### 4.1 Racing

If `U` cannot be predicted, it can be measured. Racing trains one short arm per
candidate — plus a student arm as the paired baseline — for `K` steps under
identical settings, then evaluates each resulting student on the frozen
source-free panel.

The key distinction from families 1–12 is that **the estimand does not change**.
Those families measure a proxy `X` and assume `X → U`, an extrapolation across
quantity types. Racing measures `U` itself and only shortens `K`, asking whether
`U(small K)` preserves the decision of `U(large K)` — a horizon-consistency
question in the *same* quantity, which is directly checkable.

| `K` | top-1 hit (batch 1) | top-1 hit (batch 2) |
|---|---|---|
| 2000 | 0/3 | 0/3 |
| 5000 | 3/3 | 1/3 |
| **10000** | **3/3** | **3/3** |

We adopt the conservative reading `K* = 10000`. `K = 5000` passes 3/3 in one
batch and fails in an independent one; pooling six learners gives
`U_run − U_walk = +15.61 ± 9.94 (t = 1.57)` at `K = 5000` versus `+57.14 ± 7.24
(t = 7.89)` at `K = 10000`. Within batch 1 the leader was ahead by 8.4–14.8
*episode* standard errors — an interval that looks decisive and is not, because
the relevant dispersion is across learners.

**Cost and benefit.** Choosing correctly is worth `4.95×` in median
steps-to-threshold on `hurdle` (`θ = 200`, 3/3 seeds; at `θ = 300` the wrong
source is right-censored at 100k in 3/3). Racing's theoretical minimum cost is
`3K = 30k` steps — the arms *are* the first `K` steps of the main run, so the
selected arm is retained and only three are discarded, whether the outcome is
admit or reject. Against a benefit of ≈67k steps this nets ≈`+37k`.

### 4.2 What racing measures

Racing is not measuring behavioural quality. The zero-shot behavioural ordering
on `hurdle` is `run > stand > walk`; the true utility ordering is
`run > walk > stand`. Across all **12/12** runs with `K ≥ 5000`, racing places
`walk` above `stand` — against the behavioural ordering and with the true one.

### 4.3 Admission: the same measurement decides *whether*

Selection alone is insufficient: when every source is harmful, an `argmax` still
returns one. On `crawl` the `argmax` differs on all three seeds (`stand`, `run`,
`walk` respectively) — when all utilities are negative, `argmax` is noise.

We therefore read admission off the same measurement, using the paired panel
standard error of each `U` estimate:

```
admit(T, s)  =  ∃ i :  U_i > 2 · SE_i
```

The threshold is defined relative to each estimate's own noise, so it introduces
no quantity calibrated on these targets. Adjudication over three targets × three
seeds:

`U ± paired SE` at `K = 10000`, 128 paired episodes; `*` marks `U > 2·SE`.

| target | seed | stand | walk | run | admit |
|---|---|---:|---:|---:|---|
| **crawl** (expect reject) | s1 | −44.06 ± 7.84 | −258.77 ± 18.45 | −79.17 ± 9.66 | **False** |
| | s2 | −186.76 ± 12.51 | −196.22 ± 15.21 | −181.86 ± 12.49 | **False** |
| | s3 | −162.98 ± 10.13 | −60.31 ± 13.86 | −70.35 ± 11.64 | **False** |
| **hurdle** (expect admit) | s1 | +2.57 ± 0.87\* | +62.95 ± 2.03\* | +102.19 ± 3.82\* | **True** |
| | s2 | +18.67 ± 0.90\* | +42.32 ± 2.78\* | +110.51 ± 3.70\* | **True** |
| | s3 | +7.16 ± 0.90\* | +36.67 ± 2.79\* | +81.16 ± 4.41\* | **True** |
| **slide** (expect admit) | s1 | −0.65 ± 1.27 | +48.59 ± 1.77\* | +6.67 ± 1.70\* | **True** |
| | s2 | +7.01 ± 1.80\* | +73.50 ± 2.29\* | +15.55 ± 1.22\* | **True** |
| | s3 | +2.81 ± 0.81\* | +75.83 ± 1.86\* | +26.13 ± 1.45\* | **True** |

`false_admit = 0`, `false_reject = 0`. No decision is marginal: the smallest
margin is `−59.75` (crawl s1) and `+45.06` (slide s1), i.e. 3.8×–12.7× the
threshold itself. This matters because the rule performs nine tests per target
without family-wise correction; the conclusion does not rest on borderline cells.

As a by-product this raises `crawl`'s label from a single seed to three,
with 9/9 significantly negative — the sign is learner-stable here, in contrast
to `door`.

### 4.4 Dose exit (an engineering baseline, not a contribution)

A constant 0.5 source mass for the whole run is harmful once the student
overtakes the source. On `slide`, exiting the source at 30k relative to injecting
throughout yields `+631.8 ± 9.3` at 100k (3/3), moving the endpoint from 293.3 to
929.1 and shrinking across-learner dispersion by ~8×.

**We do not claim this as a contribution.** Constant dosing is an artefact of our
own equal-dose controlled design; the original PTF formulation already decays its
transfer weight. Fixing it restores expected behaviour rather than adding a
mechanism. We include it because the end-to-end system needs *some* exit rule and
this one is verified.

---

## 5. The End-to-End Decision Chain

### 5.1 Chain

```
stage 1   racing            4 arms (3 sources + student) × K = 10000
stage 2   decision (auto)   admit = ∃i U_i > 2·SE_i ;  source = argmax_i U_i
stage 3   main training     admit  → inject the chosen source, hard-exit at 30k
                            reject → student-only
```

Stage 2 is executed by a script reading only the frozen racing output, with no
human input, and its output was committed before any main training started:

```
crawl  s1/s2/s3   REJECT
hurdle s1/s2/s3   ADMIT  run
slide  s1/s2/s3   ADMIT  walk
```

All **9/9** decisions agree with the independently established ground truth for
these targets.

### 5.2 Results, under both cost accountings

Arms: **A** = scratch; **B** = zero-cost-signal selection with injection
throughout (the zero-shot displacement `argmax`, i.e. what one would use absent
racing); **C** = the chain above. Source-free return at 100k, `sd` across
learners.

| target | A | B | C | C−A | C−B |
|---|---:|---:|---:|---|---|
| hurdle | 387.4 ± 63.1 | 479.1 ± **331.8** | **840.4 ± 11.4** | +492.0 / +498.6 / +368.2 | +271.6 / +739.8 / +72.4 |
| slide | 792.4 ± 167.3 | 293.3 ± 16.3 | **929.1 ± 20.5** | +15.6 / +348.4 / +46.0 | +648.7 / +638.4 / +620.1 |
| crawl | 960.2 ± 22.9 | 809.2 ± 73.1 | 960.2 ± 22.9 (≡A) | — | +66.3 / +149.3 / +237.4 |

Under **step-aligned accounting** the chain beats scratch on both targets where
it admits (3/3 each) and beats the zero-cost baseline on all three (3/3 each).
On `crawl` the chain reduces to scratch by construction, so `C vs A` is excluded
from the criterion; the informative comparison there is `C vs B`, which isolates
**admission**.

**The unfavourable accounting.** The chain additionally spends 40k steps on
racing in this implementation. Charging it in full and comparing the chain at
matched *total* interaction (using its 50k checkpoint as a conservative lower
bound against scratch at 100k):

| target | C−A at matched total interaction | wins |
|---|---:|---|
| hurdle | +353.8 / +386.7 / +199.4 (mean **+313.3**) | **3/3** |
| slide | −422.8 / −11.4 / −322.1 (mean **−252.1**) | **0/3** |
| crawl | −55.8 / +22.7 / −29.4 (mean −20.9) | 1/3 |

The advantage survives full cost charging on `hurdle` only. On `slide` the
scratch baseline is a late-blooming learner (259 at 50k → 792 at 100k) and the
chain's gain is in the **endpoint and its variance**, not in speed. On `crawl`
correct rejection still costs the racing budget, so the chain is necessarily
slightly worse than pure scratch — a consequence stated in the pre-registration
before the data existed, not an after-the-fact reading.

**We therefore claim higher and more stable final performance, not improved
sample efficiency.** Sample-efficiency advantage holds on one of three targets.

**What each target actually tests.** On `hurdle` and `slide` the zero-cost signal
happens to pick the same source racing picks, so `C vs B` there isolates the
*exit rule*, not selection. Only on `crawl` does `C vs B` isolate *admission*.
The value of selection is established separately (§4.1, `4.95×`).

A robust side effect: the chain removes the instability that is specific to
source-injected arms. On `hurdle`, arm B has `sd = 331.8` across learners because
one seed collapses to 111.3 — the −84% drawdown documented for constant-dose
source arms — whereas the chain has `sd = 11.4`. This comparison was not
pre-registered and is reported as descriptive.

---

## 6. Related Work

*This section requires a systematic literature search before submission; the
project has not performed one, and no priority claim is made here.*

The transfer mechanism builds on Policy Transfer Framework (Yang et al., 2020),
whose option-value and termination formulation we retain in part and whose
linearly decayed transfer weight anticipates our exit rule (§4.4). The learner is
FastTD3. Transferability metrics have been reported as fragile under systematic
benchmarking even in supervised classification, a setting far simpler than ours;
our result is the RL analogue and is empirical, not a formal impossibility
theorem. Racing is related to bandit-style arm selection, differing in that each
arm is a full learner run and the reward is a delayed, source-free evaluation.

---

## 7. Limitations

Stated as a numbered list rather than folded into discussion.

1. **Cross-task sample-efficiency gain rests on a single target.** `hurdle`
   confirms 4.38×/3.59× early speedup; `slide` is refuted (three thresholds give
   median speedups 0.851/0.627/0.758 and scratch overtakes at 100k, 792.4 vs
   293.3). The end-to-end chain wins on `slide` at the endpoint, not in speed.
2. **The `hurdle` speedup decays to 1.24× at 100k**, and the training
   instability (−66% / −84% drawdowns) is specific to source arms; scratch shows
   none across three seeds and six evaluation points.
3. **Constant dosing is our own design artefact**, not a property of PTF. The
   exit rule that fixes it is an engineering baseline (§4.4).
4. **The sign of `U` is not learner-invariant** on all targets (`door`: 18/18
   negative, then 2/9 positive). Per-seed `U` values are not reproducible ground
   truth: same-protocol reruns show `|ΔU|` median 24.23, max 43.78, against
   effect sizes of −7 to −43. Signs and orderings are usable; point values are not.
5. **Admission does not raise the ceiling on bad targets.** It spends `N×K` steps
   to avoid catastrophic negative transfer; on `crawl` this nets −20.9.
6. **All three decision targets have known ground truth.** This work therefore
   *tests a decision rule on known ground* rather than discovering new facts.
   Generalisation requires a prospective test on a target whose labels are unknown.
7. **Single batch of three learner seeds** for every adjudication except racing's
   `K*`, which has two independent batches. A 3/3 result in one batch has been
   overturned by an independent batch before in this project.
8. **No task is solved that prior methods could not solve.** `hurdle` is
   unsolved by TD-MPC2 (64.68 against a 700 bar) but FastTD3 alone solves it.
9. **One morphology, one learner, three locomotion sources.** Manipulation
   targets are outside the scope of the progress-based analyses in §3.2, whose
   notion of progress is forward displacement.
10. **Racing remains an outer-loop procedure**, not an in-algorithm component;
    the implemented cost (`4K`) exceeds the theoretical minimum (`3K`) because
    continuation from racing arms is not implemented.

---

## 8. Conclusion

The decision "which source, or none" cannot be read off any cheap function of the
task pair in the spaces we examined — and two of those failures are structural,
not empirical: identical specifications with different utilities, and an empty
feasible threshold interval. What remains is to measure, and measurement turns
out to be affordable: ten thousand steps per candidate recovers the correct
choice, and the same measurement decides whether to transfer at all. Assembled
into an automatic chain, this yields higher and more stable final performance
than either a scratch baseline or the zero-cost heuristic it replaces — with the
honest qualification that once the measurement is charged in full, the advantage
survives on one of three targets.

---

## Appendix A. Provenance of every quantitative claim

| Claim | Frozen output |
|---|---|
| twelve families / seven spaces | `docs/impossibility_characterization_of_transfer_prediction_20260730.md` |
| `slide`/`stair` byte-identical reward; `+56.95` vs `+0.19` | `docs/data/stair_bac_gate_v1/stair_bac_gate_v1_results.json`; `docs/data/slide_generalizability_v1/results.json` |
| empty threshold interval; displacements | `docs/data/progress_screen_v1/{probe,results}.json` |
| sibling asymmetry `+15.40` / `−20.79` | `docs/data/sibling_source_gate_v1/sibling_source_gate_v1_results.json` |
| sign non-invariance on `door` | `docs/data/racing_reject_door_v4/results.json` |
| `K*` and 12/12 discrimination | `docs/data/racing_min_horizon_v1/{compressed_lr,correct_lr}/results.json` |
| selection value `4.95×` | `docs/data/hurdle_selection_value_v1/results.json` |
| admission (`false_admit/false_reject`) | `docs/data/racing_admission_v1/results.json` |
| dose exit `+631.8 ± 9.3` | `docs/data/slide_hard_exit_v1/slide_hard_exit_v1_results.json` |
| `hurdle` speedup and decay | `docs/data/hurdle_speedup_v1/hurdle_speedup_v1_results.json` |
| `slide` speedup refutation | `docs/data/slide_speedup_v1/slide_speedup_v1_results.json` |
| end-to-end, both accountings | `docs/data/endtoend_v1/{decisions,results}.json` |

## Appendix B. Reproduction

All decision rules were committed to version control before the corresponding
data existed; the commit hashes are recorded in each experiment's pre-registration
document under `docs/experiments/`. Training entry point is
`fasttd3_ptf/official_fasttd3_ptf/train_ptf.py`; evaluation is
`scripts/p0_evaluator.py`, which constructs no source, option, or admission
component and is therefore source-free by construction.
