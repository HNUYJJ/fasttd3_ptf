# Research Pair Task Board

## Active

| ID | Owner | Status | Question | Output |
| --- | --- | --- | --- | --- |
| S2-V2-CHECK | claude_reviewer | done | Verify the `anchor_xattn` implementation is internally consistent before any GPU run. | 25 pytest passed incl 4 anchor_xattn tests (query-purity, skips-self-attn, cross-schema readout load) |
| S2-V2-RC | human_pi | approved | Decide whether to spend GPU on the v2-c pure robot-query experiment. | Approved by PI; v2-c parallel track greenlit |
| RC-S2-V2C-SRC | claude_reviewer | running | Train reach v2-c source (pure robot-query readout). | tmux reach_v2c, GPU0, exp reach_znative_xattn_d128_s1, ~11.8 it/s; watch eval>=6500@80k |
| RC-S2-V2C-PUSH | claude_reviewer | pending | push v2-c pilot (100k seed 1) once reach v2-c competent. | gated on RC-S2-V2C-SRC; pilot-first per Codex, escalate to 3-seed/200k only if promising |

## Candidate Follow-ups

| ID | Priority | Task | Why It Matters |
| --- | --- | --- | --- |
| S2-V2C | high | Prefer v2-c `anchor_xattn`: skip token-mixing self-attn and let a pure robot token query the entity set. | Most direct test of Claude/Codex's shared diagnosis that v1 failed because the robot query was already contaminated before readout. |
| S2-V2A | low | Keep v2-a attention masking as a fallback design only. | More complex to specify correctly and easy to create partial leakage or brittle masks. |
| S2-V2B | low | Keep v2-b robot independent path as a fallback design only. | Cleaner than masks but more code surface than v2-c for similar information gain. |
| S2-E1 | medium | Draft a trainable-E warm-start run card using the slice reach teacher. | Backup if frozen single-source E remains bad even after pure robot-query readout. |
| S2-D1 | medium | Draft a reach+push multi-task/co-training E plan with an explicit alignment objective. | Tests whether the learned-z idea needs a trained object context and cross-task alignment, not just a pooling fix. |
| S2-PKG | high | Preserve a pivot path to slice-adapter PTF on `package`. | The headline task is package/door; push is now mainly a mechanism sanity check. |
| S2-S1 | medium | Define success-rate metrics beside return for push/package. | Makes claims less reward-shaping-dependent. |

## Accepted Findings

| Finding | Action |
| --- | --- |
| A2 frozen reach-E + no teacher stayed bad (`final-5 ~= -271`), so the frozen mean-pool reach encoder is itself a poor push obs front-end. | Stop treating teacher distillation as the dominant explanation for the original mean-pool failure. |
| Anchored v1 (`pool="anchor"`) stayed bad across 3 seeds (`mean ~= -427`), despite a competent anchored reach source. | Do not spend more GPU extending v1. Treat readout-only anchoring as insufficient. |
| Mechanistic diagnosis: v1 anchors only after the Transformer, but self-attn has already mixed object information into the robot token. | If testing thesis B again, use a design that keeps the robot query pure before cross-attention. |
| The reviewer response artifact is missing from `rounds/step2_review_001/`; the actionable reviewer content currently lives in `dialogue.md`. | Record the missing artifact in the Codex response and avoid pretending the raw file exists. |

## Run Cards

### RC-S2-V2-CHECK: CPU-only implementation sanity

- **Purpose:** Confirm `anchor_xattn` is wired consistently before any expensive run.
- **Commands, no GPU training:**
  ```bash
  python -m pytest tests/test_anchor_pool.py tests/test_entity_encoder.py -q
  python scripts/step2/analyze_ab.py --window 5
  ```
- **Fallback smoke if `pytest` is unavailable in a torch-capable project shell:**
  ```bash
  python - <<'PY'
  import torch
  from fasttd3_ptf.ptf.entity import SCHEMAS, EntityObsEncoder
  kw = {"n_heads": 2, "n_layers": 1, "ff": 32, "use_hypernet": True,
        "add_type_embedding": True, "pool": "anchor_xattn"}
  enc = EntityObsEncoder(SCHEMAS["h1hand-push-v0"], d_model=32, encoder_kwargs=kw)
  z = enc(torch.randn(4, SCHEMAS["h1hand-push-v0"].obs_dim))
  assert z.shape == (4, 32)
  print("anchor_xattn smoke ok")
  PY
  ```
- **Checks:** CLI accepts `--ptf-entity-pool anchor_xattn`; `EntityEncoder(pool="anchor_xattn")` skips token-mixing self-attn; cross-schema `encoder.*` load includes `readout.*`; tests explicitly cover `anchor_xattn`, not only `anchor`.
- **Pass signal:** CPU tests pass, or the fallback smoke passes in a torch-capable shell while a proper pytest-capable dev env is queued; task-board evidence matches current logs.
- **Fail signal:** Tests only cover `anchor`, `anchor_xattn` load/smoke is untested, or CLI/encoder choices diverge.

### RC-S2-V2C-SRC: train reach source with pure robot-query readout

- **Do not launch without human_pi approval.**
- **Hypothesis:** A reach actor trained with `pool="anchor_xattn"` remains competent in-domain, so any later push failure is about transfer rather than reach source quality.
- **Command template:**
  ```bash
  CUDA_VISIBLE_DEVICES=<gpu> python -m fasttd3_ptf.official_fasttd3_ptf.train_ptf \
    --env-name h1hand-reach-v0 \
    --exp-name reach_znative_anchorxattn_d128_s1 \
    --seed 1 \
    --total-timesteps 125000 \
    --eval-interval 5000 \
    --save-interval 5000 \
    --no-compile \
    --ptf-entity-encoder \
    --ptf-entity-d-model 128 \
    --ptf-entity-pool anchor_xattn \
    --ptf-entity-no-compile-encoder
  ```
- **Success signal:** reach eval enters the same band as prior z-native sources (`>=6500` by 80k or `>=7000` by 125k).
- **Stop/fail signal:** reach cannot exceed roughly `5000` by 80k, indicating v2-c damaged in-domain source learning.

### RC-S2-V2C-PUSH: push z-PTF pilot with pure robot-query frozen E

- **Do not launch until RC-S2-V2C-SRC has a competent checkpoint and a matching source-bank yaml.**
- **Hypothesis:** If readout contamination caused v1 failure, then frozen `anchor_xattn` E should improve push over mean/anchor v1 and at least trend toward scratch.
- **Pre-run config:** Copy `configs/source_banks/h1hand_push_reach_anchor_znative.yaml` to an `anchorxattn` variant and point it at the selected reach `anchor_xattn` checkpoint.
- **Command template:**
  ```bash
  CUDA_VISIBLE_DEVICES=<gpu> python -m fasttd3_ptf.official_fasttd3_ptf.train_ptf \
    --env-name h1hand-push-v0 \
    --exp-name push_anchorxattn_zptf_s1 \
    --seed 1 \
    --total-timesteps 100000 \
    --eval-interval 10000 \
    --save-interval 25000 \
    --no-compile \
    --ptf-source-bank configs/source_banks/h1hand_push_reach_anchorxattn_znative.yaml \
    --ptf-entity-encoder \
    --ptf-entity-d-model 128 \
    --ptf-entity-pool anchor_xattn \
    --ptf-entity-load-from models/<reach_anchorxattn_checkpoint>.pt \
    --ptf-entity-freeze \
    --ptf-entity-no-compile-encoder
  ```
- **Success signal:** by 80k-100k, eval is clearly above anchored v1 and mean A, ideally crossing `0` or trending toward scratch.
- **Stop/fail signal:** eval remains around `-300` or worse through 80k-100k, or diagnostics still show object domination / teacher near-constant behavior.
- **Escalation:** Only if seed 1 is promising, extend to 200k and add seeds 2-3. Otherwise pivot to S2-E1/S2-D1/S2-PKG.

## Done

| ID | Owner | Result |
| --- | --- | --- |
| BOOT-1 | codex_engineer | Created the shared collaboration protocol, task board, dialogue log, and orchestration script. |
| S2-C1 | codex_engineer | Reconciled the available Claude summary into accepted findings, v2-c prioritization, CPU checks, and gated GPU run cards. |
| S2-A2 | human_pi + codex_engineer | A2 completed: frozen reach-E without teacher remained poor, supporting channel 1 (bad frozen representation) as the main failure. |
| S2-P1 | claude_reviewer + codex_engineer | Anchored v1 implemented and tested, but push 3-seed result was negative; readout-only anchoring is not sufficient. |
| S2-R1 | claude_reviewer | Adversarial review done (`rounds/step2_review_001/claude_reviewer_response.md`). Key: failure is REPRESENTATIONAL not pooling (A2 −271 + v1 ≈−425 jointly isolate the frozen encoder); v2-c predicted ≈ v1; **trainable warm-start E is the decisive next test** (isolates the freeze assumption); dilution "75%" metric is confounded (needs push-trained-E control); reach→push transfer signal itself is weak (C only +38 over scratch). |
