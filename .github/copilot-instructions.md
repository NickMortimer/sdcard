# Copilot Instructions for `sdcard`

## Goal
Maintain a reliable, low-risk Python CLI for SD card import workflows.

## Core Principles
- Prefer minimal, surgical changes over broad refactors.
- Fix root causes, not surface symptoms.
- Keep CLI behavior stable unless explicitly asked to change UX.
- Preserve existing option names/flags and default behavior.

## Code Style
- Follow existing project style and naming patterns.
- Avoid one-letter variables except loop indices.
- Do not add inline comments unless requested.
- Keep functions focused; avoid introducing extra abstraction for small fixes.

## CLI & UX (Typer/Rich)
- Keep command output concise and operator-friendly.
- Preserve emoji/status conventions already used in this project.
- For Rich tables, maintain stable column order and avoid noisy redraw behavior.
- If adding statuses, ensure they are deterministic and easy to parse visually.

## Safety for Import Operations
- Treat copy/move/overwrite paths as high-risk areas.
- Default to safe behavior when uncertain (no destructive action by default).
- Do not change overwrite semantics unless explicitly requested.
- Keep cross-platform behavior (Linux/Windows) intact.
- Re-importing the same card should be idempotent and quiet when files are unchanged.
- Report conflicts only when an incoming file would overwrite an existing completed file with different content.
- If a previous transfer left partial/incomplete destination files, allow overwrite/retry without conflict noise.

## Rclone Copy/Move Semantics
- Follow official rclone behavior for `copy` and `move` commands: https://rclone.org/docs/
- `rclone copy` copies source files to destination and does not remove source files.
- `rclone move` moves source files to destination and removes source files after successful transfer.
- Use modification time and file size as the default file equality/comparison criteria.
- Do not require MD5/checksum-based comparison for normal import conflict decisions unless explicitly requested.
- Keep `--update` enabled for import transfers so destination-newer files are not overwritten.
- Only enable `--check-first` when the user explicitly requests prechecks.
- Remember `--check-first` can increase memory usage because transfer backlog is built before transfer starts.
- Preserve transfer-safety flags already used by this project unless a change is explicitly requested.
- Re-import workflow expectation: unchanged files should pass without complaint, true content-overwrite risk should be surfaced, and partial-file retries should overwrite cleanly.

## Performance & Concurrency
- Avoid changing process/thread orchestration unless required for the bug/feature.
- Prefer incremental updates to worker/state tracking rather than full rewrites.
- Ensure output parsing is resilient to minor log format variations.

## Validation Expectations
- After edits, run targeted checks first:
  - static errors on changed files
  - relevant tests in `tests/`
- If no relevant tests exist, add only focused tests near changed behavior.
- Do not fix unrelated failing tests unless asked.

## File Scope Discipline
- Touch only files necessary for the requested outcome.
- Do not rename files, commands, or public symbols unless explicitly requested.
- Do not add new dependencies unless there is a clear need.

## Communication
- Summarize changes briefly with file paths and impact.
- Call out any assumptions and risks clearly.
- Suggest a practical next validation step after implementation.
