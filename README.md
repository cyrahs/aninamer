# Aninamer

Aninamer now runs as a backend service with a FastAPI API, a static review UI,
and a background worker.

- `python -m aninamer` starts the FastAPI API and the background worker.
- `python -m aninamer.api` starts the API only.

The worker owns directory discovery and processing. Frontends do not submit arbitrary paths.

## Runtime model

- The worker scans configured watch roots, creates jobs, builds rename plans, and optionally applies them.
- The API exposes job history, review snapshots, operations, safety summaries,
  job requests, runtime info, and status overviews.
- Planned and applied move items are stored as database rows with per-item status. `plan` and `rollback` JSON artifacts are not used.
- Apply result summaries are stored in PostgreSQL as internal JSONB and are not exposed by the API.

## UI

- `GET /ui` serves the static review workbench.
- The UI stores the API token in browser local storage and calls `/api/v1/*`
  with `Authorization: Bearer <api.token>`.
- Current UI surface: job list, review snapshot, source files, episode targets,
  subtitle links, TMDB context, mapping evidence, operations, safety summary,
  explicit rollback eligibility checks, TMDB-ID rematch previews, rematch commit,
  manual episode/subtitle variant corrections, and all-for-apply validation for
  proposed plans, first-pass apply execution for validated plans, and
  first-pass withdraw-to-review execution after an explicit eligibility check.

## API

Public endpoint:

- `GET /healthz`
- `GET /livez`

`GET /livez` is independent from scan staleness. It returns `503` only when the
worker is not running or every configured watch root has reported a transient
filesystem failure for two consecutive discovery scans. Storage outage and
recovery transitions are sent through the configured notification webhook.

Protected endpoints:

- `GET /api/v1/runtime`
- `GET /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/review`
- `GET /api/v1/jobs/{job_id}/operations`
- `GET /api/v1/jobs/{job_id}/operations/{operation_id}`
- `GET /api/v1/jobs/{job_id}/operations/{operation_id}/items`
- `GET /api/v1/jobs/{job_id}/safety`
- `POST /api/v1/jobs/{job_id}/rematch-preview`
- `POST /api/v1/jobs/{job_id}/rematch-commit`
- `POST /api/v1/jobs/{job_id}/corrections`
- `POST /api/v1/jobs/{job_id}/validate`
- `POST /api/v1/jobs/{job_id}/apply`
- `POST /api/v1/jobs/{job_id}/rollback-eligibility`
- `POST /api/v1/jobs/{job_id}/withdraw`
- `POST /api/v1/job-requests`
- `GET /api/v1/job-requests/{request_id}`
- `GET /api/v1/status`

All `/api/v1/*` endpoints require:

- `Authorization: Bearer <api.token>`

`GET /api/v1/status` is the frontend summary endpoint. It returns:

- `summary`
- `pending_items`
- `failed_items`

Notifications are sent as outgoing webhooks by the worker. When enabled, the worker POSTs to:

- `<notifications.base_url>/api/v2/notifications/webhook`

Request headers:

- `Authorization: Bearer <notifications.bearer_token>`
- `Content-Type: application/json`

Webhook body is fixed:

```json
{"markdown":"...","image_url":"","disable_web_page_preview":true,"disable_notification":false,"pin":false}
```

If `[notifications]` is missing or incomplete, service startup still succeeds and notifications
are recorded internally with delivery status `disabled`.

## Configuration

Service runtime is configured with `config.toml`. Use [config.toml.example](config.toml.example) as the starting point.

Supported sections:

- `[database]`
- `[tmdb]`
- `[openai]`
- `[notifications]`
- `[api]`
- `[worker]`
- `[[watch_roots]]`

TMDB and OpenAI credentials now live in `config.toml` under `[tmdb]` and `[openai]`.
Service startup no longer depends on `TMDB_API_KEY`, `OPENAI_API_KEY`, or `OPENAI_MODEL`
being present in the environment.

Config loading still does not read `.env` files.

## Docker

Build:

```sh
docker build -t aninamer .
```

Default container entrypoint starts the service launcher:

```sh
docker run --rm \
  -v "/path/to/config.toml:/app/config.toml:ro" \
  -v "/path/to/downloads:/input" \
  -v "/path/to/library:/output" \
  -p 8091:8091 \
  aninamer
```
