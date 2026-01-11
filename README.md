# Aninamer

This image runs the CLI as `python -m aninamer`.

## Environment variables

Required for `plan`, `run`, and `monitor`:
- `TMDB_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Optional:
- `OPENAI_BASE_URL` (override API base URL)
- `OPENAI_REASONING_EFFORT` (forwarded to OpenAI responses)

`apply` does not require network credentials.

## Volume mounts

Recommended container paths:
- `/input` (series directory, mount read-only if possible)
- `/output` (output root, writable)
- `/logs` (log + plan artifacts, must be outside `/input` and `/output`)

The rename plan stores absolute paths. Run `plan` and `apply` with the same mounts.

## Docker usage

Build the image:

```sh
docker build -t aninamer .
```

Plan (dry-run by default):

```sh
docker run --rm -it \
  --user "$(id -u)":"$(id -g)" \
  -e TMDB_API_KEY=... \
  -e OPENAI_API_KEY=... \
  -e OPENAI_MODEL=... \
  -v "/path/to/series:/input:ro" \
  -v "/path/to/output:/output" \
  -v "/path/to/logs:/logs" \
  aninamer plan /input --out /output --log-path /logs
```

Apply a plan:

```sh
docker run --rm -it \
  --user "$(id -u)":"$(id -g)" \
  -v "/path/to/series:/input" \
  -v "/path/to/output:/output" \
  -v "/path/to/logs:/logs" \
  aninamer apply /logs/plans/<plan>.rename_plan.json
```

Plan + apply in one step:

```sh
docker run --rm -it \
  --user "$(id -u)":"$(id -g)" \
  -e TMDB_API_KEY=... \
  -e OPENAI_API_KEY=... \
  -e OPENAI_MODEL=... \
  -v "/path/to/series:/input" \
  -v "/path/to/output:/output" \
  -v "/path/to/logs:/logs" \
  aninamer run /input --out /output --log-path /logs --apply
```

## Monitor mode

The `monitor` command watches an input directory for new series folders and automatically processes them. This is useful for continuous automation (e.g., processing downloads as they arrive).

### How it works

1. On first run, monitor records all existing directories as a **baseline** (skipped by default).
2. It periodically scans for new directories not in the baseline.
3. New directories enter a **pending** state and must remain unchanged for a settle period (default 15 seconds) to ensure downloads are complete.
4. Once settled, the directory is planned and optionally applied.
5. State is persisted to a JSON file, surviving restarts.

### Monitor options

| Option | Default | Description |
|--------|---------|-------------|
| `--apply` | off | Apply renames (otherwise plan-only) |
| `--once` | off | Run one iteration and exit |
| `--interval` | 60 | Seconds between scan iterations |
| `--settle-seconds` | 15 | Directory must be unchanged for N seconds before processing |
| `--state-file` | `<log-path>/monitor_state.json` | Path to state file |
| `--include-existing` | off | Process existing directories instead of only new arrivals |
| `--two-stage` | off | Use two-stage moves with staging temp dir |
| `--tmdb` | - | Force a specific TMDB TV id for all series |
| `--max-candidates` | 5 | Max TMDB candidates for LLM selection |
| `--max-output-tokens` | 2048 | Max output tokens for mapping LLM |
| `--allow-existing-dest` | off | Allow pre-existing destinations |

### Docker example (plan-only, continuous)

```sh
docker run -d --name aninamer-monitor \
  --restart unless-stopped \
  --user "$(id -u)":"$(id -g)" \
  -e TMDB_API_KEY=... \
  -e OPENAI_API_KEY=... \
  -e OPENAI_MODEL=... \
  -v "/path/to/downloads:/input:ro" \
  -v "/path/to/library:/output" \
  -v "/path/to/logs:/logs" \
  aninamer monitor /input --out /output --log-path /logs
```

### Docker example (with apply, continuous)

```sh
docker run -d --name aninamer-monitor \
  --restart unless-stopped \
  --user "$(id -u)":"$(id -g)" \
  -e TMDB_API_KEY=... \
  -e OPENAI_API_KEY=... \
  -e OPENAI_MODEL=... \
  -v "/path/to/downloads:/input" \
  -v "/path/to/library:/output" \
  -v "/path/to/logs:/logs" \
  aninamer monitor /input --out /output --log-path /logs --apply
```

### Docker example (single run, include existing)

Process all existing directories once and exit:

```sh
docker run --rm -it \
  --user "$(id -u)":"$(id -g)" \
  -e TMDB_API_KEY=... \
  -e OPENAI_API_KEY=... \
  -e OPENAI_MODEL=... \
  -v "/path/to/downloads:/input" \
  -v "/path/to/library:/output" \
  -v "/path/to/logs:/logs" \
  aninamer monitor /input --out /output --log-path /logs \
    --once --include-existing --apply
```

### Graceful shutdown

The monitor handles `SIGTERM` and `SIGINT` gracefully, completing the current operation before exiting. This ensures Docker stop commands work cleanly:

```sh
docker stop aninamer-monitor
```

### State file

The state file (`monitor_state.json`) tracks:
- **baseline**: directories present on first run (skipped unless `--include-existing`)
- **pending**: directories waiting to settle
- **planned**: directories with generated plans (awaiting apply)
- **processed**: successfully processed directories
- **failed**: directories that encountered errors

To reprocess a failed directory, remove it from the `failed` set in the state file and restart the monitor.
