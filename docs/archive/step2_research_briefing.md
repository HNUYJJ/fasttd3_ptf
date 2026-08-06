# Step-2 Cross-Task Obs Unification — Research Briefing & Review Request

> **Purpose of this document.** We (a human researcher + an AI coding agent) built and ran an
> experiment for a cross-task transfer idea in RL. The naive MVP **failed clearly**. We want a
> second AI (you) to act as a critical research collaborator: **review the design and the
> implementation, challenge our root-cause analysis, find flaws/bugs we missed, and propose
> better fixes.** Be skeptical and concrete. The document is self-contained — you do not have
> repo access, so everything you need (including key code) is inlined.

---

## 0. TL;DR

- **Setting:** FastTD3 (vectorized distributional TD3) + PTF (Policy Transfer Framework) on
  HumanoidBench `h1hand` tasks. Goal: cross-task skill transfer; headline target = `package`.
- **Idea under test ("step 2"):** replace hand-coded per-source observation *slice adapters*
  with a **single learned encoder** that maps any task's raw obs → a fixed-width embedding `z`,
  so a source policy can consume the *target* task's obs directly (no hand-coded adapter).
- **MVP:** train a `reach` source policy *on `z`*; reuse its **frozen** encoder as the obs
  front-end for the target `push`, with the reach actor as a teacher reading the same `z`.
- **Result (push, 200k steps, seed 1):**
  | arm | final-5-eval mean return | peak |
  |---|---|---|
  | **A** z-PTF (frozen reach-encoder + z-native reach teacher) | **−386** | −147 |
  | **B** scratch (no transfer) | **+472** | +564 |
  | **C** slice-adapter reach PTF (old hand-coded approach) | **+510** | +660 |
  - **A fails badly** (deeply negative throughout, never climbs). B & C succeed; the old
    slice-adapter transfer (C) is even slightly ahead of scratch (B).
- **Our root cause (please challenge):** the **frozen, reach-only encoder is a bad obs
  representation for `push`** — `push`'s box ("object") token type was never trained (reach has
  no manipulable object), mean-pooling dilutes the well-trained robot/hand/goal channels with
  the box's random-feature channel, and the frozen encoder can't be reshaped by the target. So
  the `push` agent is partly "box-blind."

**What we want from you:** §11 has the explicit questions.

---

## 1. Background: FastTD3 + PTF

- **FastTD3:** off-policy actor–critic. Twin **distributional (C51-style) critics**, clipped
  double-Q, 128 parallel envs, batch 32768, `num_updates=2` per env step, AMP bf16. (Note:
  `torch.compile` is force-disabled whenever the PTF logic is active, because PTF adds dynamic
  option/source-bank control flow.)
- **PTF (Policy Transfer Framework):** a set of **frozen source teacher policies** plus a
  learned **OptionModule** (per-option Q-value `Q_o` + a termination probability `β`) and an
  **OptionSelector** (call-and-return). At each step the agent either *executes/distills* a
  selected source option or picks a **null option** (no transfer). Transfer is a **masked action
  distillation** loss on the actor, gated by `λ(t)·(1−β)` (λ decays over training). Each source
  is bridged to the target task by a **hand-coded obs adapter** (slice/remap target obs → the
  source's expected obs layout) + an action adapter + an action mask.
- **Known pain point (the motivation for step 2):** these per-source obs slice adapters are
  task-specific and, if mis-specified, **silently corrupt** the distillation (wrong obs → wrong
  teacher actions, no crash). The project wants to *learn* the cross-task obs mapping instead.

HumanoidBench `h1hand` detail that matters: **all tasks share the identical robot** (a humanoid
with dexterous hands; **61 actuators**; proprioception = `qpos[0:76] + qvel[0:75] = 151` dims,
always the first 151 entries of the obs). Tasks differ only in appended object/goal state.

---

## 2. The step-2 idea

**Vision (from the human researcher):** "Unify the obs dimensions of all tasks so source and
target policies are trained on the *same* representation; then at transfer time the target
task's obs can be fed directly into the source actor, eliminating obs-dimension mismatch."

**The anchor that should make it principled:** every `h1hand` task's obs begins with the same
151-dim robot proprio block, which the tokenizer always maps to the same "robot" token. So a
*shared* encoder maps "robot in pose X" to a consistent region of `z` regardless of task — the
shared bridge for transfer.

---

## 3. The representation: an entity-token + hypernetwork encoder

Raw obs → typed "entity" tokens → set-Transformer → pooled fixed-width `z`.

- **Schema / tokenizer (declared per task, ground-truth from the simulator):** each task's flat
  obs is partitioned into typed entity tokens. 12 entity *types* (robot, hand, goal, goal_quat,
  object, object_pos, joint1d, ball, scalar, …). Each type has a canonical feature width; every
  token is zero-padded to `TOKEN_DIM = 151` (the max, = robot width). A strict coverage check
  rejects gaps/overlaps. **Obs dim is handled by the NUMBER of tokens, not token width.**
  - `reach`: obs_dim **157**, **3 tokens** = `[robot(151), hand(3), goal(3)]`.
  - `push`:  obs_dim **163**, **4 tokens** = `[robot(151), hand(3), goal(3), object(13)]`,
    where `object` = the box's free-body state (pos 3 + quat 4 + linvel 3 + angvel 3).
- **Embedding:** a **hypernetwork** generates the per-type input-projection weights (type-
  conditioned), plus an additive learned type embedding.
- **Mixing:** an **order-invariant set Transformer** (no positional encoding; identity is
  carried by the type-conditioned projection + type embedding), `norm_first`, GELU.
- **Pooling:** **masked MEAN over tokens → LayerNorm → `z ∈ R^{d_model}`.**
- **Config used:** `d_model=128, n_layers=2, n_heads=4, ff=256, hypernet=on, dropout=0`.
- **Cross-task weight sharing:** the Transformer/hypernet/type-embedding params (`encoder.*`)
  depend only on `d_model / TOKEN_DIM=151 / n_types=12`, so the **same encoder weights load into
  any task's schema**; only the (parameter-free) `tokenizer.*` buffers are task-specific.

Encoder forward (verbatim):
```python
def forward(self, tokens, type_ids, pad_mask=None):   # tokens [B,N,F]; type_ids [N]
    x = self.embed(tokens, type_ids)        # hypernet type-conditioned projection -> [B,N,d]
    x = self.in_norm(x)                     # per-token LayerNorm (balance type scales)
    if self.add_type_embedding:
        x = x + self.type_emb(type_ids)     # [N,d] broadcast
    x = self.tf(x, src_key_padding_mask=pad_mask)   # set-transformer (permutation-equivariant)
    if pad_mask is not None:
        keep = (~pad_mask).float().unsqueeze(-1)
        z = (x * keep).sum(1) / keep.sum(1).clamp_min(1.0)
    else:
        z = x.mean(dim=1)                   # <-- masked MEAN pool over tokens
    return self.out_norm(z)
```

---

## 4. The step-2 MVP design (reach → push)

We chose a **single source = `reach`** (it shares 3 of 4 `push` tokens — robot/hand/goal; only
the box `object` is new to push; and reach's hand→goal skill is plausibly relevant to push).

Pipeline:
1. **Train a z-native `reach` source from scratch.** Run FastTD3 on `reach` with the entity
   encoder as the obs front-end (the critic trains the encoder; actor/option consume a detached
   `z`; this is the step-1 "shared encoder" wiring, DrQ-v2-style). Output: a competent reach
   actor **and** a reach-shaped encoder `E`.
   - Reach trained well through `z`: deterministic eval return **1802 → 7731** (peak ~120k
     steps), plateau ~6700–7700. So `E` learned a good reach representation. We picked the
     **125k checkpoint** as the teacher.
2. **Train the target `push` with PTF, reusing `E`.**
   - Load `E` into push's obs front-end and **FREEZE** it. (Invariant: a critic-trained `E`
     would *drift*, invalidating the z-native teacher that references the same `E`.)
   - The teacher = the z-native reach actor, consuming the **same frozen `E`'s `z`** directly —
     **no slice adapter** (the encoder *is* the adapter).
   - All `h1hand` tasks share the 61-actuator action space → **identity action adapter**; action
     **mask = arms+hands** (reach is an upper-body skill; its leg outputs would fight push's
     locomotion — and this matches the validated reach-transfer setup, holding the mask constant
     across arms A and C).

The adapter-free z-native teacher (verbatim, after hardening):
```python
@torch.no_grad()
def act(self, target_obs_raw):                 # called by PTF with the TARGET task's raw obs
    was_training = self.shared_encoder.training
    self.shared_encoder.eval()                 # deterministic teacher z
    try:
        z = self.shared_encoder(target_obs_raw.to(self.device, torch.float32))  # frozen E
    finally:
        self.shared_encoder.train(was_training)
    return self.action_adapter(self.actor(z))  # reach actor on push's z; identity adapter
```

---

## 5. Implementation + hardening

New/changed code: `ZNativeSourcePolicy` (source actor on the shared frozen `E`, encoder held by
reference, not a child module); source-bank routing of `z_native` specs; training-loop flags
`--ptf-entity-load-from` (load only `encoder.*`, keep the target's tokenizer) and
`--ptf-entity-freeze` (exclude frozen `E` from the critic optimizer + grad-clip; encode under
`no_grad`).

Before committing GPU time we ran a **14-agent adversarial code review**. It found **6
defensive bugs, all fixed**:
1. resume guard didn't re-validate `freeze` (resuming could silently un-freeze `E`);
2. no startup guard tied a z-native source to a frozen+loaded `E` (forgetting a flag → silent
   teacher corruption) — now fails loudly;
3/6. `evaluate()/render()` flipped a frozen `E` back to `train()` mode (latent, dropout>0 only);
4. teacher encode could see a different dropout draw than the actor (force-eval added);
5. a width-mismatched checkpoint could silently skip the input projection (now asserted loaded).

23 unit tests pass (incl. the cross-schema encoder load: reach's `encoder.*` → push's encoder);
a full push-z-PTF integration smoke ran clean (36/36 encoder tensors loaded, encoder frozen,
teacher bank `['reach_z','null']`). **None of the 6 bugs affected the actual experiment run**
(it used the correct flags, no resume, dropout 0).

---

## 6. The experiment (3-arm A/B; push; 200k steps; seed 1)

All three arms run through the **same** `train_ptf.py` (uncompiled) with the **same** arms+hands
action mask, so the comparison is controlled except where noted.

- **A — z-PTF:** entity encoder ON, `E` = frozen reach-125k, teacher = z-native reach (reads `z`).
- **B — scratch:** entity encoder OFF, empty source bank (no transfer). The clean floor.
- **C — slice-adapter PTF:** entity encoder OFF; source = the *old raw-obs* reach checkpoint
  bridged by a hand-coded slice obs-adapter (validated in prior experiments). The "old approach."

Intended reads: **A vs B** = does z-native transfer help vs no transfer; **A vs C** = learned
shared-`z` unification vs hand-coded slice adapter.

---

## 7. Results

Eval return (deterministic, higher = better):

| step (k) | A z-PTF | B scratch | C slice-adapter |
|---|---|---|---|
| 10  | −885 | −188 | −124 |
| 40  | −297 | −15  | +5   |
| 70  | −658 | +32  | +105 |
| 100 | −270 | +122 | +229 |
| 130 | −286 | +141 | +442 |
| 160 | −570 | +395 | +482 |
| 190 | (≈−400) | +561 | +660 |
| **final-5 mean** | **−386** | **+472** | **+510** |

- **A z-PTF is deeply negative the entire run and never climbs** (best single eval −147).
- B & C climb steadily to +500–660. **C (slice) ≈ slightly > B (scratch)** (Δ≈+38; single seed,
  within noise — i.e., the old reach transfer gives at most a small benefit here).
- A vs B Δ ≈ **−858**; A vs C Δ ≈ **−896**.

---

## 8. Our root-cause analysis (please challenge)

We attribute A's failure to the **frozen, reach-only encoder being a poor obs front-end for
push**, specifically:
1. **Untrained box channel.** The `object` (box) token type **never appears in reach**, so its
   hypernet projection + type embedding are at **random init**. Frozen, push's box state becomes
   a fixed **random-feature** projection — informative in principle but an unlearned, awkward
   representation the downstream nets must decode.
2. **Mean-pool dilution.** `z` is the mean over tokens. push has 4 tokens; the 3 well-trained
   ones (robot/hand/goal) are averaged with the 1 random one (object), so the box "noise" leaks
   into every `z` and the proprio signal is down-weighted vs reach's 3-token mean.
3. **Frozen ⇒ unfixable.** The target cannot fine-tune `E` to learn to see the box (freezing is
   required to keep the z-native teacher valid). So push is structurally **partly box-blind** —
   fatal for a task that is *about* perceiving and moving the box.

**Isolation argument:** A and B/C differ in two things at most (obs pathway; presence of a
teacher). But **C uses a reach teacher too and HELPS** (+510 > +472), so a reach teacher is *not*
harmful. The only structural thing unique to A is the **frozen reach-`E` obs pathway** ⇒ that is
the culprit, not the transfer/teacher.

---

## 8b. Empirical diagnostics (measured — added after the initial hypothesis)

We then **measured** the frozen reach-125k `E` + z-native reach teacher on 832 real push and
832 real reach observations (collected via random actions from env reset). This refines §8:

| metric | reach-z (in-domain) | push-z (target) | reading |
|---|---|---|---|
| z per-dim std | 0.59 | 0.30 | push-z ~2× lower variance |
| effective dim (participation ratio /128) | 57.5 | 38.6 | push-z compressed, **not** collapsed |
| **teacher action cross-obs std** | **0.81** | **0.30** | teacher ~2.7× **less responsive** on push-z |
| teacher saturation frac | 0.56 | 0.62 | similarly/more saturated |
| push-z OOD frac (vs reach [1,99]%) | — | 0.046 | push-z **not** globally OOD |
| box **state** drives | — | 9.9% of push-z spread | box perceivable but **weak** |
| zeroing box obs shifts z | — | 3.68 (≈ full z spread) | box token = large **constant** offset |

**Refined root cause — two channels, not one:**
1. The untrained `object` token adds a large *constant* random offset to every push-z; the box's
   *state* modulates z only ~10% → push perceives the box **weakly** (our "box-blind" claim in §8
   was too strong — it's ~10%, not 0%).
2. push-z is **compressed** (eff-dim 39 vs 58), and in that squeezed region the reach **teacher's
   action is near-constant across push states** (cross-obs std 0.30 vs 0.81). Distillation
   (arms+hands, λ·(1−β)) then drags the push actor toward an *uninformative saturated arm/hand
   action* — a **second, distinct failure channel** that plausibly explains why A goes *negative*
   (actively misled) rather than merely ≤ scratch. Consistent with C succeeding: C's raw-obs
   reach teacher reads push obs in reach's native raw layout and stays responsive.

**Self-critique (we flag against ourselves):** these obs were collected with **random actions**
(robot flailing near reset, box barely pushed), an off-distribution regime — see §8c, where the
on-policy re-measurement **overturns two of these conclusions**.

## 8c. ON-POLICY re-measurement — corrects §8/§8b

Re-measured with **on-policy** obs (B-scratch push actor drives push; z-native reach actor drives
reach; same frozen reach-125k E; 9600 obs each):

| metric | reach-z (in-domain) | push-z | random-action (§8b) said |
|---|---|---|---|
| effective dim (participation ratio /128) | 77.9 | 45.7 | 57.5 / 38.6 |
| teacher cross-obs std (arms+hands, what PTF distills) | 0.81 | 0.47 | 0.30 |
| teacher saturation (arms+hands) | 0.54 | 0.60 | — |
| push-z OOD frac (vs reach [1,99]%) | — | 0.033 | 0.046 |
| **box STATE drives push-z spread** | — | **75.4%** | **9.9%** |
| teacher action Δ when box frozen (arms+hands) | — | 0.18 | — |

**Corrections (we were wrong; flagging it):**
- ❌ **"box-blind" is FALSE.** With the box actually moving (on-policy), the box *state* drives
  **75%** of push-z variation (random actions barely moved the box → the 9.9% was an artifact),
  and the teacher's arm/hand actions react to box motion (Δ0.18). push is not blind — it is
  **box-DOMINATED**: the *arbitrary, untrained* `object` projection dominates the mean-pooled z
  and **dilutes the robot/hand/goal anchor**.
- ⚠️ **"teacher near-constant garbage" is OVERSTATED.** On-policy the teacher's cross-obs std is
  **0.47 vs 0.81** in-domain — less discriminating, *not* dead (0.30 was the low-variance artifact).
- ✅ **Robust across both measurements:** push-z is **compressed** (eff-dim 46 vs 78) and its
  variance is **dominated by an arbitrary box projection** that dilutes proprio.

**Revised mechanism — *dilution-by-domination*, not blindness:** the untrained `object` token's
arbitrary projection (a) hands the target a frozen z whose variance is mostly arbitrary box
features (hurts target RL — channel 1), and (b) dilutes the proprio the reach teacher relies on,
giving reduced-but-nonzero teacher responsiveness (channel 2). **A2 (frozen reach-E + entity ON +
NO teacher) is running to weight the two channels:** A2≈scratch ⇒ teacher-distillation is the main
culprit; A2 also-bad ⇒ the frozen box-dominated representation is itself the problem; A2 middling
⇒ both.

---

## 9. Caveats & confounds (be skeptical of us)

- **Single seed.** No variance estimate. (The project has a known determinism gap; multi-seed is
  needed for claims. The A-vs-B/C gap is huge (~850–900), so the *sign* is unlikely to be noise,
  but treat magnitudes as provisional.)
- **A-vs-C teacher confound.** A's teacher is the **z-native** reach (trained through `E`); C's
  teacher is the **raw-obs** reach (a *different* network). Both are competent reach policies,
  but they are not identical, so A vs C is "system vs system," not a single-variable comparison.
- **Missing clean ablation.** We did **not** run "frozen reach-`E` + *no* teacher" (A2), which
  would isolate the encoder handicap from the teacher entirely. (We argue C already implies it.)
- **`E` capacity / config.** `d_model=128, 2 layers` — small. Pooling is plain masked-mean (no
  CLS / no dedicated robot read-out). `torch.compile` off on the PTF path; the encoder adds a
  ~2× throughput tax.
- **Reach was trained *with* a co-trained encoder, then frozen** — so `E` is optimized for reach
  RL, not for being a general/target-friendly representation.

---

## 10. Fix directions we are weighing (critique these; propose better)

1. **Trainable warm-started `E`.** Initialize `E` from reach but **don't freeze** — let push's
   critic fine-tune it so it learns to see the box. (A trainable `E` can't host the z-native
   teacher, since `E` drifts; pair it with the slice-adapter reach teacher.) Fast, one run;
   tells us if a reach-pretrained, fine-tunable encoder can match/beat scratch.
2. **Multi-task `E` (faithful to the vision).** Pretrain `E` on reach **and** push together so
   the box token gets trained (e.g. add a self-supervised reconstruction / inverse-dynamics
   objective on push obs, or co-train), then freeze and use the z-native teacher. Keeps the
   adapter-free teacher; more engineering + compute.
3. **Better pooling.** Dedicated **robot/CLS read-out** instead of mean-pool, so the proprio
   channel is invariant to token count/content. Addresses dilution but **not** box-blindness on
   its own.
4. **Pivot.** Since C (slice-adapter) already ≥ scratch with far less machinery, possibly drop
   the learned-encoder line and put compute toward the headline (`package`) with the working
   slice-adapter PTF; keep the encoder as future work.

---

## 11. Specific questions for you (the reviewing AI)

1. **Is our root-cause correct?** Give alternative explanations for A's collapse to *negative*
   return (note B/C are positive). E.g., could the frozen `E` + distillation interact to
   actively destabilize the actor? Could `out_norm`/mean-pool produce a near-constant `z` that
   starves the critic? Could the arms+hands mask + reach z-teacher push the actor into a bad
   basin?
2. **Design flaws** in the encoder (mean-pool, hypernet, no robot read-out, `d_model=128`) or in
   the MVP framing (single source, freeze requirement, reach as the source for a locomotion-
   heavy push task)?
3. **Implementation bugs** we may have missed beyond the 6 found — especially around the
   frozen-encoder gradient/no-grad path, the z-native teacher consuming a *different* schema's
   tokenization (reach-trained actor fed push's 4-token `z`), or the distillation gating.
4. **Is "freeze the encoder for the target" fundamentally flawed**, or fixable? Which fix in §10
   is most promising, and is there a better one (e.g., partial fine-tuning, adapter on `z`,
   contrastive alignment, train `E` on all tasks first)?
5. **Is the "z-native teacher" concept sound?** Should source→target transfer instead distill
   the source into a *target-obs-conditioned* head, or align z-spaces explicitly?
6. **Better experiment design:** what ablations (A2 = frozen-E no-teacher; trainable-E; multi-
   task-E), how many seeds, what metrics (success rate vs return), to make the claim rigorous?
7. **Strategic:** given C already ≥ scratch cheaply, is the learned-encoder direction worth the
   ~2× cost and complexity for the headline goal (`package`)? What would change your answer?
8. **On §8b:** which failure channel dominates — the *compressed/box-weak representation* (hurts
   the target's own RL) or the *near-constant teacher advice* (hurts via distillation)? What is
   the single most decisive measurement or ablation to separate them (e.g. A2 = frozen-E + no
   teacher isolates channel 1; on-policy obs re-measurement removes our random-action confound)?
   And is our two-channel reading even right?

---

### Appendix: key implementation facts for review

- Reach z-native training = `train_ptf.py` on `h1hand-reach-v0`, entity encoder ON, **empty
  source bank** (FastTD3-on-z; the option machinery is inert with only a null option).
- Frozen-encoder gating in the target loop: `_train_encoder = use_entity_encoder and not
  entity_frozen`. When frozen: `E` excluded from the critic optimizer **and** grad-clip; obs
  encoded under `torch.set_grad_enabled(False)`; next-obs always under `no_grad`. The critic
  still trains its *own* head on the (fixed) `z`.
- Entity types & widths: `robot=151, robot_qpos=76, hand=3, goal=3, goal_quat=4, object=13,
  object_pos=7, joint1d=2, ball=7, scalar=1, …`; `TOKEN_DIM=151` (pad), `N_TYPES=12`.
- Action space: 61 actuators shared across all `h1hand` tasks ⇒ identity action adapter.
- `v_min/v_max` (distributional critic support) are task-specific: push uses ±1000, reach ±2000.
