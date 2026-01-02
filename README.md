# aninamer (Docker)

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
