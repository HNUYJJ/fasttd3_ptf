# Claude Reviewer Response — step2_review_001 (S2-R1)

Role: `claude_reviewer`. Adversarial scientific review of the Step-2 z-native transfer
failure analysis. Uses the **latest experimental data** (the briefing is a snapshot from
*before* A2/anchored-v1 finished): A2 done (final-5 **−271**), anchored-v1 3-seed done
(recent mean **≈ −425**), v2-c (`anchor_xattn`) reach source currently training.

---

## Verdict (one paragraph)

The *dilution-by-domination* story is directionally plausible but rests on a **confounded
metric**, and the team's own follow-ups push toward a more uncomfortable conclusion than
"fix the pooling." A2 (no teacher, **−271**) proves the frozen reach-only encoder is a bad
push front-end *by itself*; anchored-v1 (a strictly better, robot-anchored readout) lands
at **≈ −425 ≈ mean-A (−386)**, i.e. re-weighting tokens at readout **does nothing**. Read
together, the harmful variable lives in the **frozen, push-untrained representation**
(an `object` projection stuck at reach-init that the target cannot reshape), **not in how
tokens are pooled**. The most probable outcome of v2-c is therefore **another null** — it
changes pooling while leaving the frozen-untrained-box-projection root cause untouched. The
single highest-value next measurement is **trainable warm-start E** (freezing is the
load-bearing assumption — test it directly), not more frozen-pooling variants. And the quiet
red flag nobody is pricing in: **C beats scratch by only +38** (single-seed, within noise),
so even a *faithful* reach teacher gives essentially **no positive transfer** to push — which
means the bottleneck may be **reach→push transferability itself**, a source-selection problem
that no encoder change can fix.

## Strongest alternative explanations / failure modes

1. **"Box drives 75% of push-z" is a confound, not evidence of harm.** push *is* a box task;
   a *well-trained* push encoder would also have box state dominate z-variance. High box share
   ≠ dilution harm. The causal variable is that the box projection is **frozen at reach-init
   (untrained)**, not that it dominates. The diagnostics conflate "box dominates variance" with
   "box projection is bad." **Missing control:** measure box-driven variance share for a
   *push-trained* encoder. If it's also ~75%, the dilution narrative loses its main quantitative
   support.
2. **Root cause is representational, not pooling.** A2 (no teacher) still fails (−271 ≪ scratch
   +472) and anchored-v1 (better pooling) ≈ mean-A. Both isolate the *frozen encoder
   representation* as the harmful factor. Pooling/readout changes operate **downstream** of the
   frozen, untrained `object` projection — they cannot inject box information that projection
   never learned. **Prediction: v2-c ≈ v1.**
3. **Weak/absent reach→push transferability.** C (slice-adapter, a *correct* raw-obs reach
   teacher) beats scratch by only **+38** (within single-seed noise). Even a faithful reach
   teacher → ~no positive transfer to push. reach (arm-to-target) and push (whole-body box
   pushing) are far apart. If the *pair* is the problem, no encoder fix helps; this is source
   selection masquerading as representation.
4. **Distillation-into-a-bad-basin (channel 2) is real but secondary.** A2−A ≈ 115 says the
   teacher adds harm, but A2 alone is already −271, so the teacher is not the dominant term.
   Confirm via β/λ gating stats, but it is not the main lever.
5. **Critic starvation on an ill-conditioned z.** out_norm + mean-pool over a frozen,
   low-eff-dim z (≈46/128) may give the push critic a poorly-conditioned input → unstable Q and
   *negative* (not merely ≤scratch) returns. Check Q-spread/critic loss across pooling variants.

## Implementation risks / checks to run

- **Box-share control on a push-trained encoder** — the missing control for the whole dilution
  claim. Analysis-only, ~free, narrative-deciding.
- **β/λ gating + option-selection logs during A** (S2-B1): how often was the z-teacher actually
  distilled? Quantifies channel-2 magnitude directly.
- **z-conditioning panel:** per-dim std, eff-dim, **critic Q-spread/loss** for push-z under
  {frozen reach-E, anchor_xattn-E, trainable-E}. Confirms whether any pooling change improves
  conditioning at all.
- **v_min/v_max mismatch (low):** reach teacher shaped under ±2000 critic support; push uses
  ±1000. Teacher is a tanh actor (not support-bound) so likely benign — flag only.
- **anchor_xattn static coverage already verified** (25 tests incl. query-purity / skips-self-
  attn / cross-schema readout load). No further static risk there.

## Ablations ranked by information gain per unit compute

1. **Trainable warm-start E** (reach-init, **unfrozen**, push critic fine-tunes; no z-teacher,
   or slice-adapter reach teacher; arms+hands mask; 100k seed1). Tests the load-bearing *freeze*
   assumption. All three outcomes decisive: ≥scratch ⇒ freezing was the culprit, learned-z is
   salvageable (must be trainable); ≈scratch ⇒ freezing isn't the only issue, reach→push signal
   is weak; <scratch ⇒ reach features actively harmful ⇒ pivot. **Highest info/compute.**
2. **Box-share control on a push-trained E** (analysis-only, ~free). Validates or kills the
   dilution metric.
3. **Multi-task E** (reach+push co-train so the box projection is *trained*), then frozen +
   z-teacher. Faithful to the vision; tests whether a trained box projection rescues the frozen
   path. Higher compute.
4. **v2-c `anchor_xattn`** (already running, sunk-cheap). Tests pure-query pooling. **Predicted
   ≈ v1.** Let it finish for completeness; do **not** gate decisions on it producing a positive.
5. **Multi-seed mean-A / v1** (low). Sign is already unambiguous (~850 gap); seeds add rigor,
   little new information.

## One decisive next action — with the exact evidence

**Launch trainable warm-start E on push** (reach-init entity encoder, **NOT frozen**, push
critic fine-tunes it; start with **no z-native teacher** to isolate the representation; arms+
hands mask; 100k; seed1; on the next free GPU).

Exact evidence it produces — a push eval curve that **cleanly separates "frozen-ness" from
"reach-representation quality,"** the one variable A2 and anchored-v1 left entangled:
- **trainable-E reaches scratch's band (~+400+):** freezing + z-native-teacher is the problem;
  drop that construction, move to trainable-E + explicit z-alignment.
- **trainable-E ≈ scratch but no gain:** the encoder isn't the bottleneck — **reach→push
  carries no transferable signal**; for `package`, re-select a source with real skill overlap
  (likely a push/locomotion source, not reach) and switch the headline metric to **success
  rate**, not return.
- **trainable-E < scratch:** reach features are actively harmful ⇒ strong pivot signal.

This single 100k run has **strictly higher information gain than v2-c**, because v2-c only
varies pooling *within* the frozen regime that A2 already implicated.

## Strategic answer (Q7): is learned shared-z worth package compute?

**Not yet.** On current evidence learned-z has **zero positive results**, and its cheaper rival
(slice-adapter, C) shows only **noise-level** benefit (+38) on push — i.e. *neither* path has
demonstrated **positive transfer** on reach→push at all. The blocking question for `package` is
not "which encoder" but "**is there a source whose skill actually transfers to package, measured
by success rate, not shaped return**." Resolve transferability first with one well-chosen
source/target pilot + success-rate metric; only then invest in the representation that carries
it. Spending the ~2× encoder tax on `package` before that is premature.

---

## Execution status (reviewer is ahead of the run-card flow)

- **S2-R1 (this review): done.**
- **CPU/static (RC-S2-V2-CHECK): done** — 25 pytest pass incl. 4 `anchor_xattn` tests.
- **v2-c reach source (RC-S2-V2C-SRC): running** — tmux `reach_v2c`, GPU0, wandb `76pw574d`;
  watcher (tmux `push_v2c_watch`) auto-launches the push v2-c pilot at the 80k ckpt.
- **Reviewer recommendation to Codex/PI:** keep v2-c running (sunk-cheap, completeness), but
  **prioritize a trainable-warm-start-E run card (S2-E1) as the actually-decisive experiment** —
  v2-c is predicted ≈ v1, and trainable-E tests the real load-bearing assumption (freeze).

## 5–8 bullet summary (most critical conclusions)

- **The dilution-by-domination metric is confounded:** "box drives 75% of z" is expected for any
  push encoder; the real culprit is the *frozen, untrained* box projection, not its variance
  share. Needs a push-trained-encoder control.
- **The failure is representational, not pooling:** A2 (no teacher, −271) + anchored-v1 (better
  readout, ≈−425≈mean-A) jointly isolate the frozen encoder representation. Pooling changes can't
  inject box info the frozen projection never learned.
- **v2-c (`anchor_xattn`) is predicted to ≈ v1** (another null): it improves the readout but
  leaves the frozen-untrained-box-projection root cause intact. Let it finish, don't bet on it.
- **C beats scratch by only +38 (noise):** even a faithful reach teacher gives ~no positive
  transfer — a strong hint that **reach→push transferability itself** is the bottleneck (source
  selection), independent of any encoder.
- **Highest-info next experiment = trainable warm-start E** (reach-init, unfrozen, 100k seed1):
  one run cleanly separates "frozen" from "reach-representation quality," which A2/v1 left
  entangled.
- **For `package`, learned shared-z is not yet worth the compute:** neither learned-z nor
  slice-adapter has shown *positive* transfer on push; resolve transferability (success-rate
  metric, well-matched source) before investing the encoder's 2× cost on the headline.
- **Channel-2 (teacher distillation) is real but secondary** (A2−A≈115 vs A2 itself −271);
  confirm with β/λ gating logs, don't treat it as the main lever.
