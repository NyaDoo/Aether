# Video Generation Surfaces

Aether exposes three independent video client contracts. They share the
`video_tasks` table, the polling worker and the billing pipeline, but each keeps
its own request/response shape, route classification and plan kinds.

| Surface | `api_format` | Client routes |
| --- | --- | --- |
| OpenAI (Sora) | `openai:video` | `POST /v1/videos`, `GET/DELETE /v1/videos/{id}`, `POST /v1/videos/{id}/{cancel,remix}`, `GET /v1/videos/{id}/content` |
| Gemini (Veo) | `gemini:video` | `POST /v1beta/models/{model}:predictLongRunning`, `GET /v1beta/operations/{id}`, `POST .../{id}:cancel` |
| Doubao (Volcengine Ark) | `doubao:video` | `POST /v3/contents/generations/tasks`, `GET /v3/contents/generations/tasks`, `GET/DELETE /v3/contents/generations/tasks/{id}`, `GET /v3/contents/generations/tasks/{id}/content` |

## Prerequisite

Video tasks are only polled when the gateway runs with Rust as the source of
truth:

```
--video-task-truth-source-mode rust-authoritative
# or AETHER_GATEWAY_VIDEO_TASK_TRUTH_SOURCE_MODE=rust-authoritative
```

The default is `python-sync-report`, under which no poller runs and tasks stay
in `submitted` forever. This applies to all three surfaces.

## Doubao (Volcengine Ark)

### Endpoint configuration

- `api_format`: `doubao:video`
- `base_url`: `https://ark.cn-beijing.volces.com/api`
- Auth: bearer token (`Authorization: Bearer <ARK_API_KEY>`), same as the
  OpenAI surface

The base URL is appended to verbatim with the same path clients call, so the
upstream URL is always `<base_url>/v3/contents/generations/tasks`. Nothing is
inferred or repaired: a base that already ends in `/v3` yields `/v3/v3/...` and
Ark answers `404`, leaving the misconfiguration visible rather than silently
rewritten. Endpoint `custom_path` is rejected for video creation because a
create-only path does not define safe GET/DELETE resource templates.

Clients point their Ark SDK at `https://<aether-host>/v3`. Every task read,
download, delete and list request must carry the caller's Aether API key; the
gateway uses its resolved user identity to prevent cross-tenant task access.
Provider-side GET/DELETE requests are separately authenticated with the
configured Ark bearer key. That credential is rehydrated from provider config
at runtime and is not persisted inside task metadata snapshots.

### Request modeling

The gateway parses only what it needs and passes everything else through
verbatim, so new Ark parameters keep working without a gateway change.

| Field | Why the gateway reads it |
| --- | --- |
| `model` | model mapping and routing |
| `content[]` first `text` entry | stored as the task prompt for display |
| `ratio` | billing dimension (`aspect_ratio`) |
| `resolution` | billing dimension |
| `duration` | billing dimension (`duration_seconds`) |
| `seed` | preserved in task GET/list responses |
| `frames` | preserved in task GET/list responses; mutually exclusive with `duration` |
| `framespersecond` | preserved in task GET/list responses |
| `callback_url` | rejected with `400` |

Everything else — including `generate_audio`, `watermark`, `camerafixed`, and
reference image/video/audio entries with their `role` values — is forwarded
untouched. Ark response `duration` is projected as a JSON string, matching the
official GET/list schema.

Ark also accepts the older prompt-suffix syntax (`--rt 16:9 --dur 5`). Top-level
fields win; suffixes are only parsed as a fallback when the corresponding
top-level field is absent, and only to recover billing dimensions.

### `callback_url`

Rejected with `400`. The gateway owns task state, so an upstream callback would
bypass it and leak the upstream task id. Silently stripping the field would be
worse: the client would wait for a callback that never arrives.

Use polling, or the download endpoint below.

### Task identifiers

Clients receive a gateway-local id shaped like Ark's (`cgt-<uuid>`); the
upstream id is never exposed. This keeps clients that validate the id format
working while preserving failover.

### Listing

`GET /v3/contents/generations/tasks` is answered from gateway state, not
proxied. Proxying would return every task owned by the shared provider key
across all tenants. Supported query parameters: `page_size`, `page_num`,
`filter.status` (`queued` / `running` / `succeeded` / `failed` / `cancelled`),
`filter.model`, and `filter.task_ids` (repeated or comma-separated). Task IDs
are filtered after ownership checks, before pagination. Unknown filters are
ignored rather than rejected. `filter.model` is an exact match against the
request-side model/Ark endpoint ID; the `model` returned in each item remains
the provider-resolved model name and version.

### Downloading

`GET /v3/contents/generations/tasks/{id}/content` is an Aether extension, not an
Ark route. Ark's `content.video_url` is a signed URL that expires within a day,
so the extension proxies the bytes for callers that do not want to consume the
signed URL directly. For native Ark compatibility, a succeeded task's normal
GET/list body still contains `content.video_url`; non-succeeded tasks never
emit stale content URLs. The OpenAI projection never exposes the Ark URL and
uses `/v1/videos/{id}/content` exclusively. The upstream credential is
deliberately not attached to an asset request, because the URL already carries
its own signature.

Asset downloads require HTTPS. Before connecting, the gateway resolves every
A/AAAA record, rejects the whole answer set if any address is private or
reserved, and pins the validated addresses for the connection. Redirects are
not followed. Because remote proxy, tunnel, and browser transports cannot honor
that DNS pin, an asset plan using one of them fails closed; task polling and
cancellation still use the configured provider proxy normally.

`?variant=last_frame` serves the last-frame image when the task was created with
`return_last_frame`.

### Cancel and delete

Ark exposes a single `DELETE`, which cancels a running task and removes a
finished one. There is no separate cancel route. Cancelled tasks remain
queryable for the provider retention window; explicitly deleted tasks read back
as `404`.

### Billing

Ark bills by tokens, unlike the per-second OpenAI and Gemini video surfaces.
`usage.completion_tokens` is recorded as `output_tokens`; `input_tokens` is
derived as `total_tokens - completion_tokens`, and `total_tokens` is preserved.
Token pricing can therefore charge input and output independently. `resolution`
and `duration` are also recorded as dimensions for tiered rules.

## Multi-node deployment

Supported for all three surfaces:

- The poller is a cross-node singleton keyed on the instance id, so only one
  node polls.
- Due tasks are claimed with a database lease to prevent duplicate work.
- The database is authoritative; each node's task store is a local cache that is
  hydrated on read miss.
- If a stored snapshot cannot be deserialized, the task is rebuilt from its
  stored columns plus the provider transport.
- Cross-instance identity forwarding is accepted only with a short-lived HMAC
  proof. Set the same 32-byte-or-longer
  `AETHER_GATEWAY_INTERNAL_FORWARD_SECRET` on every gateway instance (a secure
  shared `JWT_SECRET_KEY` is used as fallback). Production deployments should
  use a dedicated random secret and carry internal administrator proofs only
  over mTLS or an equivalently trusted encrypted network because administrator
  proofs are not currently bound to a destination instance.

During a rolling upgrade, Doubao tasks are only serviceable by upgraded nodes.
OpenAI and Gemini tasks are unaffected, so no downtime is required.
