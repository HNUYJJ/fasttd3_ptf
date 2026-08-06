# Codex-Claude Research Pair

This folder is the shared workspace for a two-agent research workflow. The goal
is not to make the VS Code chat panels click each other. The goal is to make
Codex and Claude exchange explicit, auditable artifacts: prompts, responses,
task decisions, experiment evidence, and follow-up actions.

## Roles

- `codex_engineer`: implementation, repo navigation, experiment wiring,
  diagnostics, tests, and reconciliation into concrete patches or commands.
- `claude_reviewer`: independent critical review, alternative explanations,
  methodology critique, ablation prioritization, and paper-style argument
  checks.
- `human_pi`: final decision-maker for expensive experiments, claims, and any
  broad direction change.

## Operating Rules

- One writer at a time. The active implementation agent may edit files; the
  reviewer should produce critique and suggested changes first.
- Every nontrivial claim should point to at least one artifact: code path, log,
  experiment table, paper section, or diagnostic output.
- Every agent response should end with concrete next actions, ranked by
  expected information gain.
- Expensive training runs require a short run card before launch: command,
  hypothesis, expected duration, success/failure signal, and stop condition.
- Keep raw agent output in `rounds/<round_id>/`; summarize decisions in
  `dialogue.md` and update `research_tasks.md`.

## Minimal Workflow

1. Prepare prompts for a round:

   ```bash
   python scripts/research_pair.py prepare \
     --round-id step2_review_001 \
     --brief docs/step2_research_briefing.md \
     --task "Critically review the Step-2 z-native transfer failure analysis."
   ```

2. Send the Claude reviewer prompt:

   ```bash
   python scripts/research_pair.py run \
     --role claude_reviewer \
     --prompt docs/agent_collab/rounds/step2_review_001/claude_reviewer_prompt.md \
     --out docs/agent_collab/rounds/step2_review_001/claude_reviewer_response.md
   ```

3. Ask Codex to reconcile Claude's critique into repo actions:

   ```bash
   python scripts/research_pair.py run \
     --role codex_engineer \
     --prompt docs/agent_collab/rounds/step2_review_001/codex_engineer_prompt.md \
     --out docs/agent_collab/rounds/step2_review_001/codex_engineer_response.md
   ```

4. Update `research_tasks.md` with accepted actions and record the round in
   `dialogue.md`.

## Safety Notes

- `research_pair.py run` executes external CLIs. Use `doctor` first to check
  what is installed.
- Keep reviewer turns read-only unless you intentionally want an implementation
  pass.
- Do not paste secrets into prompts. Prompts are stored on disk.
