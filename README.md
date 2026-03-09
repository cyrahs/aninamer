# Aninamer

Aninamer now runs as a backend service with a FastAPI API and a background worker.

- `python -m aninamer` starts the FastAPI API and the background worker.
- `python -m aninamer.api` starts the API only.

The worker owns directory discovery and processing. Frontends do not submit arbitrary paths.

## Runtime model

- The worker scans configured watch roots, creates jobs, builds rename plans, and optionally applies them.
- The API exposes job history, job requests, runtime info, and a status overview endpoint for pending and failed items.
- Internal artifacts for `plan`, `result`, and `rollback` are stored in PostgreSQL as JSONB and are not exposed by the API.

## API

Public endpoint:

- `GET /healthz`

Protected endpoints:

- `GET /api/v1/runtime`
- `GET /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
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
{"markdown":"...","disable_web_page_preview":true,"disable_notification":false}
```

If `[notifications]` is missing or incomplete, service startup still succeeds and notifications
are recorded internally with delivery status `disabled`.

## Configuration

Service runtime is configured with `config.toml`. Use [config.toml.example](config.toml.example) as the starting point.

Supported sections:

- `log_path`
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
  -v "/path/to/logs:/logs" \
  -p 8091:8091 \
  aninamer
```
