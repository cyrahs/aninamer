# tests layout

Tests are grouped by feature area to keep discovery predictable:

- `tests/apply/`: `aninamer.apply`
- `tests/cli/`: CLI commands and monitor behavior
- `tests/episode_mapping/`: mapping prompt/output parsing and orchestration
- `tests/openai/`: OpenAI client and opt-in real-LLM integration smoke tests
- `tests/plan/`: plan builder and plan I/O
- `tests/tmdb/`: TMDB client, resolver, and query-cleaning behavior
- `tests/utils/`: scanner/subtitles/name-clean/json helpers and logging instrumentation
- `tests/real_cases/`: fixture-based real-case plan assertions

Data files that only serve one test area should live under that same directory
(for example `tests/real_cases/data/`).

## Naming

Use `test_<module>_<focus>.py` so names stay consistent and searchable.

- prefer `core`, `flow`, `validation`, `defaults`, `integration`, `compat`
- avoid ad-hoc suffixes like `codex`, `unit`, `module`, `extra`
- keep one dominant focus per file; split instead of chaining with `and_*`
