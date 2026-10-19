# Scene Analyzer — scene understanding with a Vision-Language Model

Upload an image and get back a structured analysis: objects with attributes and bounding boxes,
spatial/semantic relationships, activities, scene context (type, indoor/outdoor, time of day,
mood, likely purpose), visible text, and a natural-language caption and description. The VLM is
**Claude**, called through the Anthropic API.

```
image ──► preprocess ──► Claude (vision + JSON-schema output) ──► validate ──► clean-up ──► scene graph
          (orient, flatten,   structured outputs guarantee         Pydantic    ids, boxes,    nodes + edges
           resize, encode)    schema-valid JSON                                dangling refs
```

| Project component | Where |
|---|---|
| Image preprocessing / input handling | `app/preprocess.py` |
| VLM inference | `app/backends.py` (`ClaudeBackend`) |
| Objects, attributes, relationships, activities, context, text | `app/schema.py` + the prompt in `app/analyzer.py` |
| Structured representation + scene graph | `app/schema.py`, `build_scene_graph()` |
| Web app / API / CLI | `app/main.py`, `app/static/index.html`, `app/cli.py` |
| Evaluation | `evaluate.py` |

## Quick start

```bash
cd scene_analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then put your key in .env:  ANTHROPIC_API_KEY=sk-ant-...
python -m uvicorn app.main:app --port 8000
```

Open <http://127.0.0.1:8000> — drop, choose or paste an image. Hover an object or relationship to
highlight its box. Interactive API docs are at `/docs`.

The first time, choose **Create account** and make yourself a username and password; accounts are kept in
`data/scene_analyzer.db`. Each person signs in with their own account.

**API** — sign in once, keep the cookie, then post images:

```bash
curl -c jar -H 'Content-Type: application/json' -d '{"username":"me","password":"my long password"}' \
     http://127.0.0.1:8000/api/login
curl -b jar -F file=@photo.jpg -F focus="read any signs" http://127.0.0.1:8000/analyze
```

**CLI**

```bash
python -m app.cli photo.jpg --out photo.scene.json
```

The key is read from the environment or `.env` and is never sent to the browser. Run like this, the
server listens on localhost and anyone who can reach it may create an account. Before putting it on the
internet, read **Deploying** below, because every analysis spends your API credits.

## Output

`POST /analyze` returns `{analysis, scene_graph, meta}`:

- `analysis.caption`, `analysis.description`
- `analysis.objects[]`: `id`, `label`, `category`, `prominence`, `attributes[{name,value}]`, `bbox` (`[x1,y1,x2,y2]`, 0–1, top-left origin, or `null`), `confidence`
- `analysis.relationships[]`: `subject_id`, `predicate`, `object_id`, `kind` (spatial / functional / interaction / part_of / comparative), `confidence`
- `analysis.activities[]`: `description`, `actor_ids`, `target_ids`, `confidence`
- `analysis.context`: `scene_type`, `environment`, `time_of_day`, `weather_or_lighting`, `atmosphere`, `probable_purpose_or_event`, `clues[]`
- `analysis.visible_text[]`, `analysis.uncertainties[]`
- `scene_graph`: `nodes` (objects + flattened attributes) and `edges` (relationships) — ready for networkx / Neo4j / a graph viewer
- `meta`: model, latency, token usage, image sizes sent, and post-processing `warnings`

## Design notes

- **Structured outputs, not prompt-and-parse.** The Pydantic schema is converted to a JSON schema and sent as `output_config.format`, so the response is schema-valid JSON. (Forced `tool_choice` is rejected by Sonnet 5.5, so tool-use JSON was not an option.) Constraints the schema language can't express (unique ids, box ranges, references that resolve) are repaired in `postprocess()`, and each repair is reported in `meta.warnings`.
- **Bounding boxes are approximate.** Claude can localize objects, but it is not a detector; treat boxes as visual aids, not measurements. Boxes are normalized, so they stay correct when the image is downscaled. Pixel-valued boxes are auto-normalized.
- **Image handling.** EXIF rotation applied, transparency flattened to white, longest side capped at `MAX_IMAGE_EDGE` (default 1568 px — raise it for tiny text or fine detail), PNG kept for PNG inputs, animated images use the first frame.
- **People.** The prompt tells the model to describe people by appearance and action, not to identify them from faces or infer sensitive attributes.
- **Privacy.** Images are sent to the Anthropic API for analysis. Do not use it for images you are not allowed to share with a third-party service.
- **Model.** Default `claude-sonnet-5-5`; set `CLAUDE_MODEL` to use another vision-capable Claude model. Each request is one image plus a few thousand output tokens.

## Evaluation

```bash
# your own annotations (see examples/annotations.example.json for the format)
python evaluate.py --images data/imgs --annotations data/ann.json --out eval_out --judge

# COCO val2017 (images dir + annotation files from cocodataset.org)
python evaluate.py --images val2017 --coco-instances annotations/instances_val2017.json \
    --coco-captions annotations/captions_val2017.json --limit 100 --judge
```

Outputs `eval_out/results.json` (per image), `eval_out/summary.md`, and a `cache/` so re-runs don't
re-bill the API.

| Metric | Meaning |
|---|---|
| `object_precision / recall / f1` | Canonical label sets (synonym map + head noun: "golden retriever" → dog). On COCO, precision counts only predictions that map to a COCO class, because COCO labels just 80 classes. |
| `scene_type_match`, `environment_match` | Soft match on scene type; exact match on indoor/outdoor |
| `activity_recall` | Reference activities covered by a predicted activity or the caption |
| `relationship_pair_recall / triple_recall` | Reference (subject, predicate, object) found with the right objects / right objects *and* predicate |
| `caption_token_f1`, `caption_rougeL` | Best over reference captions. Crude: they punish correct but differently worded text |
| `judge_accuracy / completeness / hallucination_free` | `--judge`: Claude scores 1–5 while looking at the image. Reference annotations are treated as incomplete hints |

Caveats: COCO ground truth omits many real objects, so precision is a lower bound and recall is the
more trustworthy object metric (`--min-area-frac` ignores tiny instances). The judge is the same model
family as the analyzer, so pass `--judge-model` with a different model to reduce self-preference bias.
There are no public ground-truth labels for relationships/attributes at this granularity, so for those
use your own annotations or the judge.

## Accounts and deploying

Everyone signs in with **their own username and password**; nobody uses yours. People create their own
account on the sign-in page. Because every analysis spends your Claude API credits, a public deployment
is locked down by default:

- **Invite code.** Creating an account needs `REGISTRATION_CODE`, which you give to the people you want
  to allow. (`OPEN_REGISTRATION=1` lets anyone sign up; only use it with low limits.)
- **Your admin account.** `ADMIN_USERNAME` / `ADMIN_PASSWORD` create your account on first start. At
  `/admin` you can see who has signed up and how much each person has used, disable or delete accounts,
  and reset a forgotten password (it shows a temporary one, once).
- **Limits.** 20 analyses per person per day (`USER_DAILY_LIMIT`), 300 per day across everyone
  (`GLOBAL_DAILY_LIMIT`), 10 a minute per person, 4 at once. Photos that fail to analyze don't count.
  Also set a spending limit in your Anthropic console as a backstop.
- **Passwords** are stored as salted scrypt hashes, never in plain text. Login cookies are signed,
  HttpOnly and Secure (on https). Repeated wrong passwords lock the account and the address out for a while.
  Changing or resetting a password signs that person out of every device.
- **The Docker image refuses to start** (and tells you why) if `SECRET_KEY`, `DATABASE_URL`, an invite
  code, or a proper admin password is missing.

### Where accounts live: `DATABASE_URL`

Accounts and usage counts need a database that survives restarts. A free web service's disk is wiped on
every deploy and restart, so a SQLite file there would lose every account. Use a hosted Postgres and put
its connection string in `DATABASE_URL` (`postgres://` and `postgresql://` both work).

- **Neon or Supabase** (both have free tiers; check their current limits) keep the data as long as your
  account lasts. Create a project, copy the connection string, and paste it in.
- **Render's own free Postgres** expires after 30 days, so it is only good for a quick trial.

The tables are created automatically.

### Render

1. Push this folder to a GitHub repository.
2. In the Render dashboard choose **New > Blueprint** and pick the repository. `render.yaml` defines the
   service. Render asks for `ANTHROPIC_API_KEY`, `DATABASE_URL`, `ADMIN_USERNAME`, `ADMIN_PASSWORD` and
   `REGISTRATION_CODE`; `SECRET_KEY` is generated for you.
   (If the service already exists, add the same variables under its **Environment** tab and save.)
3. When the deploy finishes, open the `onrender.com` address, sign in as your admin user, and give your
   friends the address and the invite code. They choose **Create account** and pick their own login.

Free instances sleep when idle, so the first request after a pause is slow. Use a paid instance for a
demo you need ready at any moment.

### Any Docker host

```bash
docker build -t scene-analyzer .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... -e DATABASE_URL='postgresql://...' \
  -e SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
  -e REGISTRATION_CODE='an invite code' -e ADMIN_USERNAME=owner -e ADMIN_PASSWORD='a long passphrase' \
  scene-analyzer
```

The image listens on `$PORT` when the host sets it (Render, Railway, Cloud Run) and on 8000 otherwise.
Keep the key in the host's secret store rather than a plain environment variable where one is offered.

### Without Docker

```bash
pip install -r requirements.txt
REQUIRE_AUTH=1 DATABASE_URL=... SECRET_KEY=... REGISTRATION_CODE=... ADMIN_USERNAME=owner ADMIN_PASSWORD=... \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
```

Put a TLS-terminating proxy (Caddy, nginx) in front of it. Without TLS, passwords cross the network in the clear.

### Things to know

- **Request time.** An analysis takes roughly 10 to 30 seconds. If your host cuts requests off sooner,
  lower `ANTHROPIC_TIMEOUT_S` and `ANTHROPIC_MAX_RETRIES` or pick a host with longer timeouts.
- **Memory.** Uploads are downscaled before use and JPEGs are decoded at reduced size. Images over
  `MAX_IMAGE_MEGAPIXELS` (50) are rejected; lower that on very small instances.
- **Photos are not stored.** Only usernames, password hashes and per-day usage counts are kept.
- **One instance.** The per-minute limits live in memory, so with several instances each enforces its own;
  the daily limits are in the database and are shared.
- **Fonts.** The page loads two fonts from Google Fonts, so visitors' browsers contact Google. Delete the
  Google Fonts `<link>` tags in the pages under `app/static/` to avoid that; system fonts are used instead.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Runs without an API key or network: a mock backend covers preprocessing, post-processing, the API,
and the evaluator; a mocked HTTP transport checks the exact request `ClaudeBackend` sends and how API
errors map to HTTP statuses.

## Configuration

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | |
| `MAX_IMAGE_EDGE` | `1568` | px, longest side sent to the model |
| `MAX_UPLOAD_MB` | `20` | |
| `MAX_OUTPUT_TOKENS` | `8192` | raise if you see "output was truncated" on very busy scenes |
| `REQUIRE_AUTH` | off (Docker image: on) | checks that the settings below are safe and refuses to start otherwise |
| `DATABASE_URL` | `sqlite:///./data/scene_analyzer.db` | required when `REQUIRE_AUTH=1`; use a persistent Postgres |
| `SECRET_KEY` | random per start locally | 32+ characters; signs login cookies |
| `REGISTRATION_CODE` | none | invite code needed to create an account |
| `OPEN_REGISTRATION` | off | let anyone create an account |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | none | create your admin account on first start (password 10+ characters) |
| `USER_DAILY_LIMIT` | `20` | analyses per person per UTC day, `0` = unlimited |
| `GLOBAL_DAILY_LIMIT` | `300` | analyses per day across everyone, `0` = unlimited |
| `RATE_LIMIT_PER_MIN` | `10` | analyses per person per minute, `0` = off |
| `GLOBAL_RATE_LIMIT_PER_MIN` | `60` | analyses per minute across everyone, `0` = off |
| `MAX_CONCURRENT` | `4` | analyses running at once on one instance |
| `SESSION_DAYS` | `7` | how long a sign-in lasts |
| `COOKIE_SECURE` | auto | force the Secure flag on the login cookie |
| `MAX_IMAGE_MEGAPIXELS` | `50` | larger images are rejected |
| `ANTHROPIC_TIMEOUT_S` / `ANTHROPIC_MAX_RETRIES` | `90` / `2` | worst-case request time is about timeout x (retries + 1) |
| `PORT` | `8000` | set by most hosts; used by the Docker image |
