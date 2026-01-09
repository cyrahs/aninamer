````markdown
# AGENTS.md — aninamer (Codex)

This repository builds **aninamer**, a safe CLI tool that renames **anime episode video files** and their **external subtitle files** using **TMDB metadata** and **LLM-based mapping**.

Codex should implement the project incrementally, one step at a time, with **pytest coverage** for each step.

---

## 0) Core principles (non-negotiable)

1. **Safety first**
   - Never delete files.
   - Default behavior is **dry-run**.
   - Always generate a **rename plan** (manifest) before applying changes.
   - Validate everything before applying: schema, bounds, collisions, path traversal.

2. **LLM does mapping; code does enforcement**
   - LLM decides which files are regular episodes / OVA(OAD) specials and their `SxxExx`.
   - Code performs deterministic rename path construction, collision handling, and file ops.

3. **Touch as little as possible**
   - Only rename:
     - episode video files
     - subtitle files explicitly associated with those episodes
   - Leave untouched: fonts, images, NFO, archives, OP/ED, PV, trailers, extras, and any unmapped files.

4. **No episode titles in filenames**
   - Episode filename format is strictly:  
     `{series_zh_cn} S{season:02d}E{episode:02d}{ext}`  
     or multi-episode: `{series_zh_cn} S{season:02d}E{e1:02d}-E{e2:02d}{ext}`

5. **Naming language**
   - Series title must use **Simplified Chinese** from TMDB (`language=zh-CN`) when available.

6. **Subtitles (Chinese-only assumption)**
   - Add suffix based on variant:
     - Simplified → `.chs`
     - Traditional → `.cht`
     - Unknown Chinese → `.chi`
   - Subtitle output examples:
     - `…/S01/动画名 S01E01.chs.ass`
     - `…/S00/动画名 S00E02.cht.srt`

7. **OVA/OAD**
   - Included and placed in `S00/`.
   - Prefer matching by TMDB Season 0 (specials) title/overview that mention OVA/OAD.
   - If no explicit OVA/OAD hints: assume **local OVA/OAD order matches TMDB specials order**.

8. **No access to .env files**
   - Never read, modify, or access `.env` files.
   - These files contain sensitive credentials and configuration that must remain untouched.

---

## 1) Repository structure (target)

Codex should keep responsibilities separated and code testable.

Suggested modules:

- `aninamer/scanner.py`
  - Scan filesystem, collect candidate videos/subtitles, assign stable integer IDs.
- `aninamer/tmdb_client.py`
  - TMDB API wrapper (search tv, tv details, seasons, specials). Must be mockable.
- `aninamer/tmdb_resolve.py`
  - Resolve TMDB TV ID from dirname with optional LLM assist.
- `aninamer/prompts.py`
  - Prompt builders (tmdb id selection; episode mapping).
- `aninamer/llm_client.py`
  - LLM interface (injectable/mocked in tests). No provider specifics in core logic.
- `aninamer/mapping.py`
  - Parse/validate LLM mapping JSON and convert to internal model.
- `aninamer/subtitles.py`
  - Detect `.chs/.cht/.chi` deterministically (filename hints first; content sampling fallback).
- `aninamer/plan.py`
  - Build rename plan, compute destinations, detect collisions, write `rename_plan.json`.
- `aninamer/apply.py`
  - Apply rename plan safely, write results + rollback plan.
- `aninamer/cli.py`
  - CLI entrypoints (`plan`, `apply`).

Artifacts:
- `rename_plan.json`
- `rename_result.json`
- `rollback_plan.json`
- `.cache/tmdb/…` (optional; must be configurable)

---

## 2) Coding standards

- Language: **Python 3.10+** (prefer 3.11+ if available).
- Use `pathlib.Path` for all path operations.
- Type hints everywhere (`mypy`-friendly).
- Prefer `dataclasses` or `pydantic` models for structured data.
- All I/O isolated behind small functions to allow mocking.
- Do not embed network calls directly in core logic.
- Use `logging` (no prints in library code; CLI may print summaries).

### Error handling
Define small custom exception types, e.g.:
- `AninamerError` (base)
- `PlanValidationError`
- `LLMOutputError`
- `TMDBError`
- `ApplyError`

Errors should be actionable and include context (which file id/path, which season/episode, etc.).

---

## 3) Testing rules (pytest)

- Use **pytest**.
- Tests must be **offline**: never hit TMDB or LLM network endpoints.
- Mock TMDB client responses and LLM outputs.
- Prefer `tmp_path` fixtures for filesystem behavior.
- Include tests for:
  - strict schema validation
  - episode bounds validation vs TMDB episode counts
  - collision detection
  - subtitle suffix detection
  - “only rename episodes + mapped subs” guarantee
  - dry-run outputs do not change the filesystem
  - apply produces rollback plan

Command:
- `pytest -q`

---

## 4) Interfaces and contracts

### Candidate file IDs
Scanner assigns stable IDs per run:
- videos: `[{id, rel_path, ext, size_bytes, (optional) duration_sec}]`
- subtitles: `[{id, rel_path, ext, size_bytes}]`

IDs are used in LLM I/O to save tokens.

### TMDB resolve (LLM-assisted)
Given:
- dirname (cleaned + raw)
- top N TMDB search candidates (id, name, first_air_date, origin_country, etc.)
LLM returns:
- strict JSON: `{ "tmdb": <int> }`
The code validates returned TMDB id is among candidates (unless explicitly allowed by config).

### Episode mapping output schema (compact)
LLM must return **only** rename targets:

```json
{
  "tmdb": 123,
  "eps": [
    { "v": 1, "s": 1, "e1": 1, "e2": 1, "u": [101, 102] }
  ]
}
```

* `v` = video id
* `u` = subtitle ids associated with this video (optional; may be empty)
* `s` = season (0 allowed)
* `e1/e2` = episode range (single if e1==e2)

Code must validate:

* ids exist & types match (videos vs subtitles)
* season/episode bounds:

  * if `s==0`: `1..special_count`
  * else: `1..season_episode_count[s]`
* no two entries map to same destination
* LLM output contains no extra keys

### Destination path construction (deterministic)

Do not accept destination paths from LLM.

* Root folder:

  * `{series_zh_cn} ({year}) {tmdb-{tmdb_id}}/`
* Season folder:

  * `S{season:02d}/`
* Video filename:

  * `{series_zh_cn} S{season:02d}E{ep:02d}{ext}`
  * range: `{series_zh_cn} S{season:02d}E{e1:02d}-E{e2:02d}{ext}`
* Subtitle filename:

  * same base + `.{chs|cht|chi}.{sub_ext}`

Sanitize series name for filesystem safety and avoid path traversal.

---

## 5) CLI behavior (v1 target)

* `aninamer plan <series_dir> --out <output_root> [--tmdb <id>] [--dry-run]`

  * scans candidates
  * resolves TMDB id (LLM assist allowed)
  * fetches TMDB season counts and specials list
  * calls LLM mapping
  * produces `rename_plan.json` + preview output

* `aninamer apply <rename_plan.json>`

  * executes renames/moves
  * writes `rename_result.json` + `rollback_plan.json`

Defaults:

* Dry-run unless `apply` is invoked.
* Never modify files outside the given output root.

---

## 6) Implementation workflow for Codex (how to work)

For each step:

1. Implement a small, testable unit (pure function first).
2. Add or update pytest tests that describe expected behavior.
3. Run `pytest -q` until green.
4. Avoid introducing unrelated refactors.

When a step requires LLM:

* Implement prompt builder and strict JSON parser/validator.
* In tests, mock LLM output strings.

When a step requires TMDB:

* Implement a mockable `TMDBClient` interface.
* In tests, provide fake responses.

---

## 7) Definition of Done (per step)

A step is done when:

* It has pytest coverage.
* It is deterministic under test.
* It respects safety invariants (no deletes, no touching untouched resources).
* It produces useful errors on invalid inputs.
* It does not require real TMDB/LLM network calls during tests.

---

## 8) Notes for future versions (do not implement unless asked)

* Heuristic pre-mapping (currently not desired).
* Multi-series batch mode.
* Subtitle content language detection beyond Chinese-only assumption.
* More sophisticated duplicate-release selection.

---

```