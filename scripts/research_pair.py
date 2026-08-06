#!/usr/bin/env python3
"""Prepare and optionally run Codex/Claude research-pair turns.

The script writes prompts and responses under docs/agent_collab so a research
discussion can be replayed later. It is intentionally small: the durable unit is
a file-backed round, not an opaque chat-panel transcript.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_BRIEF = Path("docs/step2_research_briefing.md")
DEFAULT_ROOT = Path("docs/agent_collab")


@dataclass(frozen=True)
class Role:
    name: str
    executable: str
    prompt_file: str
    response_file: str
    instructions: str


ROLES: dict[str, Role] = {
    "claude_reviewer": Role(
        name="claude_reviewer",
        executable="claude",
        prompt_file="claude_reviewer_prompt.md",
        response_file="claude_reviewer_response.md",
        instructions=(
            "You are Claude Code acting as an independent critical research "
            "collaborator. Be skeptical, concrete, and adversarial in the "
            "scientific sense. Your job is to challenge the root-cause "
            "analysis, find design flaws or implementation risks, rank "
            "ablations by information gain, and recommend the next decisive "
            "measurement. Do not rewrite code in this turn unless explicitly "
            "asked. Prefer evidence-backed claims and call out uncertainty."
        ),
    ),
    "codex_engineer": Role(
        name="codex_engineer",
        executable="codex",
        prompt_file="codex_engineer_prompt.md",
        response_file="codex_engineer_response.md",
        instructions=(
            "You are Codex acting as the implementation and reconciliation "
            "agent. Read the research brief and any reviewer response, then "
            "turn accepted critique into concrete repo actions: code checks, "
            "diagnostic scripts, run cards, tests, or patches. Keep changes "
            "narrow, verify them locally, and separate confirmed facts from "
            "hypotheses."
        ),
    ),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_text(path: Path, *, max_chars: int | None = None) -> str:
    text = path.read_text(encoding="utf-8")
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED]\n"
    return text


def rel(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_round_dir(root: Path, round_id: str) -> Path:
    round_dir = root / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir


def make_prompt(
    role: Role,
    *,
    task: str,
    brief_path: Path,
    extra_context: Iterable[Path],
    max_chars: int | None,
    round_dir: Path | None = None,
) -> str:
    brief_text = read_text(brief_path, max_chars=max_chars)
    extra_blocks: list[str] = []
    for path in extra_context:
        if not path.exists():
            extra_blocks.append(f"## Missing Context: {rel(path)}\n\nFile not found.")
            continue
        extra_blocks.append(
            f"## Extra Context: {rel(path)}\n\n```text\n"
            f"{read_text(path, max_chars=max_chars)}\n```"
        )

    required_output = (
        "## Required Output\n\n"
        "- Verdict in one paragraph.\n"
        "- Strongest alternative explanations or failure modes.\n"
        "- Implementation risks or checks to run.\n"
        "- Ablations ranked by information gain per unit compute.\n"
        "- One decisive next action, with the exact evidence it would produce.\n"
    )
    if role.name == "codex_engineer":
        reviewer_hint = ""
        if round_dir is not None:
            reviewer_hint = (
                "\n- If present, read and reconcile this reviewer response "
                f"before proposing repo actions: `{rel(round_dir / ROLES['claude_reviewer'].response_file)}`.\n"
                "- If that file is missing, say the reviewer turn has not run yet."
            )
        required_output = (
            "## Required Output\n\n"
            "- Accepted findings to act on.\n"
            "- Concrete repo changes or diagnostic commands.\n"
            "- Verification plan.\n"
            "- Any run card needed before GPU work.\n"
            f"{reviewer_hint}"
        )

    return "\n\n".join(
        [
            "# Research Pair Turn",
            f"Role: `{role.name}`",
            f"Generated: `{utc_stamp()}`",
            "## Role Instructions",
            role.instructions,
            "## Task",
            task.strip(),
            required_output,
            f"## Primary Briefing: {rel(brief_path)}",
            "```markdown\n" + brief_text + "\n```",
            *extra_blocks,
        ]
    )


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_version(exe: str) -> str:
    found = shutil.which(exe)
    if found is None:
        return "missing"
    for args in ([exe, "--version"], [exe, "-v"]):
        try:
            proc = subprocess.run(
                args,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = proc.stdout.strip().splitlines()
        if output:
            return f"{found} | {output[0]}"
    return found


def cmd_doctor(args: argparse.Namespace) -> int:
    root = Path(args.root)
    brief = Path(args.brief)
    print("Research-pair doctor")
    print(f"  workspace: {Path.cwd()}")
    print(f"  collab root: {root} ({'ok' if root.exists() else 'missing'})")
    print(f"  briefing: {brief} ({'ok' if brief.exists() else 'missing'})")
    if brief.exists():
        text = brief.read_text(encoding="utf-8")
        print(f"  briefing size: {len(text)} chars, {len(text.splitlines())} lines")
    for role in ROLES.values():
        print(f"  {role.executable}: {command_version(role.executable)}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    root = Path(args.root)
    brief = Path(args.brief)
    if not brief.exists():
        raise SystemExit(f"Briefing not found: {brief}")

    round_id = args.round_id or f"{utc_stamp()}_research_pair"
    round_dir = ensure_round_dir(root, round_id)
    extra_context = [Path(p) for p in args.context]

    written: list[str] = []
    for role in ROLES.values():
        prompt = make_prompt(
            role,
            task=args.task,
            brief_path=brief,
            extra_context=extra_context,
            max_chars=args.max_chars,
            round_dir=round_dir,
        )
        prompt_path = round_dir / role.prompt_file
        prompt_path.write_text(prompt, encoding="utf-8")
        written.append(rel(prompt_path))

    manifest = {
        "round_id": round_id,
        "created_utc": utc_stamp(),
        "brief": rel(brief),
        "task": args.task,
        "context": [rel(path) for path in extra_context],
        "prompts": written,
    }
    manifest_path = round_dir / "manifest.json"
    write_json(manifest_path, manifest)

    print(f"Prepared round: {rel(round_dir)}")
    for path in written:
        print(f"  wrote {path}")
    print(f"  wrote {rel(manifest_path)}")
    return 0


def run_agent(role: Role, prompt: str, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    if shutil.which(role.executable) is None:
        raise SystemExit(f"Executable not found on PATH: {role.executable}")
    if role.name == "codex_engineer":
        cmd = [role.executable, "exec", "--ephemeral", prompt]
    elif role.name == "claude_reviewer":
        cmd = [role.executable, "-p", "--output-format", "text", prompt]
    else:
        raise SystemExit(f"Unknown role: {role.name}")
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
    )


def append_dialogue(root: Path, role_name: str, prompt_path: Path, out_path: Path) -> None:
    dialogue = root / "dialogue.md"
    dialogue.parent.mkdir(parents=True, exist_ok=True)
    block = (
        f"\n## {utc_stamp()} - {role_name}\n\n"
        f"- Prompt: `{rel(prompt_path)}`\n"
        f"- Response: `{rel(out_path)}`\n"
    )
    if dialogue.exists():
        with dialogue.open("a", encoding="utf-8") as fh:
            fh.write(block)
    else:
        dialogue.write_text("# Agent Dialogue Log\n" + block, encoding="utf-8")


def cmd_run(args: argparse.Namespace) -> int:
    role = ROLES[args.role]
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        raise SystemExit(f"Prompt not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")
    out_path = Path(args.out) if args.out else prompt_path.with_name(role.response_file)

    if args.dry_run:
        print(f"Dry run for {role.name}")
        print(f"  prompt: {rel(prompt_path)}")
        print(f"  output: {rel(out_path)}")
        print(f"  executable: {command_version(role.executable)}")
        return 0

    proc = run_agent(role, prompt, timeout_sec=args.timeout_sec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(proc.stdout, encoding="utf-8")
    append_dialogue(Path(args.root), role.name, prompt_path, out_path)

    print(f"{role.name} exited with code {proc.returncode}")
    print(f"  wrote {rel(out_path)}")
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="collaboration artifact root")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check local Codex/Claude availability")
    doctor.add_argument("--brief", default=str(DEFAULT_BRIEF))
    doctor.set_defaults(func=cmd_doctor)

    prepare = sub.add_parser("prepare", help="write prompt files for a research round")
    prepare.add_argument("--brief", default=str(DEFAULT_BRIEF))
    prepare.add_argument("--round-id", default="")
    prepare.add_argument("--task", required=True)
    prepare.add_argument(
        "--context",
        action="append",
        default=[],
        help="extra file to inline into prompts; can be passed multiple times",
    )
    prepare.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="truncate each inlined file to this many characters",
    )
    prepare.set_defaults(func=cmd_prepare)

    run = sub.add_parser("run", help="execute one prepared prompt through an agent CLI")
    run.add_argument("--role", choices=sorted(ROLES), required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--out", default="")
    run.add_argument("--timeout-sec", type=int, default=1800)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
