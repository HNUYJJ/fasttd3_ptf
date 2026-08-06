# Codex Engineer Response - step2_review_001

## Source Status

Requested reviewer artifact:

`docs/agent_collab/rounds/step2_review_001/claude_reviewer_response.md`

was not present in the workspace when this response was prepared. I reconciled the
available reviewer content from `docs/agent_collab/dialogue.md`, especially the
anchored-v1 negative result and the v2 design fork recorded there. No GPU
training was launched.

## Executable Claude Suggestions

1. **Retire readout-only anchoring as a fix.**
   - Evidence: anchored v1 push remained negative across three seeds and is close
     to the mean-pool A failure.
   - Action: do not extend v1 or spend more compute on `pool="anchor"` unless a
     new diagnostic contradicts the contamination hypothesis.

2. **Treat A2 as decisive channel-isolation evidence.**
   - Evidence: frozen reach-E with no teacher stayed poor.
   - Action: the main failure is the frozen mean-pool representation itself, not
     primarily teacher distillation.

3. **If testing thesis B again, prioritize v2-c.**
   - v2-a attention masks are easy to under-specify and may leak.
   - v2-b robot-independent path is plausible but has more code surface.
   - v2-c is the cleanest diagnostic: skip token-mixing self-attention and let a
     pure robot proprio embedding cross-attend the entity set.

4. **Require CPU/static verification before any GPU run.**
   - The current code exposes `--ptf-entity-pool anchor_xattn` and implements an
     `anchor_xattn` branch, so the next engineering step is to verify tests cover
     that mode explicitly.

5. **Keep the pivot path alive.**
   - If v2-c is flat, the frozen single-source-E story is likely exhausted for
     push. Next choices should be trainable warm-start E, multi-task/aligned E,
     or the pragmatic slice-adapter package track.

## Concrete Repo Actions

Updated `docs/agent_collab/research_tasks.md` with:

- accepted findings from A2 and anchored v1;
- active CPU-only consistency task `S2-V2-CHECK`;
- v2-c GPU source and push run cards, gated by human PI approval;
- lower-priority v2-a/v2-b fallback tasks;
- pivot tasks for trainable-E, multi-task-E, and slice-adapter package.

`docs/agent_collab/dialogue.md` should be updated with a short note that this
artifact exists and that the raw Claude response file was missing.

## Recommended Next Step

Run only CPU/static checks first:

```bash
python -m pytest tests/test_anchor_pool.py tests/test_entity_encoder.py -q
python scripts/step2/analyze_ab.py --window 5
```

In this shell, `pytest` and `torch` are both unavailable (`No module named
pytest`, `No module named torch`), so use the project training/test environment
for the pytest command or the fallback smoke check recorded in
`research_tasks.md` before any GPU launch.

Before launching any GPU job, confirm that the tests include `anchor_xattn`
coverage rather than only `anchor` coverage. If the CPU check reveals missing
coverage, add the test first; do not launch a training run to debug an unpinned
architecture.

## Run Card Pointer

The full run cards are now in `docs/agent_collab/research_tasks.md`:

- `RC-S2-V2-CHECK`: CPU-only implementation sanity.
- `RC-S2-V2C-SRC`: reach source training with pure robot-query readout.
- `RC-S2-V2C-PUSH`: push z-PTF pilot using the matching frozen E.
