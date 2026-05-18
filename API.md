# TRIBE Brain Analysis API

A REST API that predicts a person's brain response to a given stimulus and returns a behavioral interpretation of that response.

**Base URL:** `https://acti.cognition.ngrok.pro`

**Auth:** HTTP Basic — `user / 67420`

**Live OpenAPI docs:** `https://acti.cognition.ngrok.pro/docs` (Swagger UI, behind the same basic auth)

---

## 1. `POST /analysis`

Run the full pipeline (brain-response prediction + behavioral analysis) on input text and return the analysis.

### Request

```http
POST /analysis HTTP/1.1
Host: acti.cognition.ngrok.pro
Authorization: Basic <base64(user:67420)>
Content-Type: application/json

{
  "text": "She walked into the empty room and froze.",
  "n_timesteps": 10,
  "view": "left",
  "cmap": "fire"
}
```

**Body (application/json):**

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `text` | string | yes | — | `minLength: 1`. The stimulus the brain is "experiencing". |
| `n_timesteps` | integer | no | `10` | `1..30`. How many predicted activation timesteps to render. |
| `view` | string | no | `"left"` | enum: `left`, `right`, `dorsal`, `ventral`, `medial_left`, `medial_right`. |
| `cmap` | string | no | `"fire"` | enum: `fire`, `hot`, `seismic`, `bwr`, `coolwarm`. |

### Response

**200 OK:**
```json
{
  "analysis": "<natural-language behavioral interpretation, ~5-10 KB of Markdown-formatted text>"
}
```

The `analysis` string is freeform Markdown. Expect headers, bullet lists, bold emphasis. Plan for ~5–10 KB per call.

### Error responses

| HTTP | When | Body |
|---|---|---|
| **401 Unauthorized** | Missing or invalid basic auth. | `{"detail": "Invalid credentials"}` |
| **422 Unprocessable Entity** | Schema validation failed (empty `text`, `n_timesteps` out of range, invalid `view`/`cmap`). | Pydantic error list under `detail`. |
| **502 Bad Gateway** | Internal pipeline failure (prediction failed, downstream analysis failed). | `{"detail": "<error message>"}` |

---

## 2. `GET /healthz`

Liveness check. No auth required.

**Response 200:**
```json
{ "status": "ok", "model_loaded": true }
```

`model_loaded` flips to `true` once the brain-encoding model has finished loading after process start (~30s).

---

## 3. Calling `/analysis`

### curl

```bash
curl -u user:67420 -X POST https://acti.cognition.ngrok.pro/analysis \
  -H "Content-Type: application/json" \
  -d '{"text": "She walked into the empty room and froze."}'
```

### Python (requests)

```python
import requests

resp = requests.post(
    "https://acti.cognition.ngrok.pro/analysis",
    auth=("user", "67420"),
    json={"text": "She walked into the empty room and froze."},
    timeout=20 * 60,  # see Performance section
)
resp.raise_for_status()
print(resp.json()["analysis"])
```

### Node (fetch)

```javascript
const res = await fetch("https://acti.cognition.ngrok.pro/analysis", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Basic " + Buffer.from("user:67420").toString("base64"),
  },
  body: JSON.stringify({ text: "She walked into the empty room and froze." }),
});
const { analysis } = await res.json();
console.log(analysis);
```

---

## 4. Performance & operational notes

- **Latency: 10–15 minutes per call** for typical inputs (a paragraph to a film-script-length passage). Most of that is brain-response prediction; the analysis step adds a few minutes. **Set client timeouts to ≥15 minutes.**
- **Concurrency: one request at a time.** Concurrent requests are queued behind a global lock inside the service. Plan to call serially or budget for additional wait time on top of the per-call latency.
- **Treat calls as async on your side.** Don't block a UI thread; surface a "processing" state to your end user.
- **No rate limiting** is currently enforced. Be considerate.
- **No idempotency keys / no cancellation.** Once a request is in flight, you cannot cancel it from the client side; just abandon the connection and the server will continue to completion.

---

## 5. Sample run

A request submitted with a 2,247-character film-script passage (Macbeth's "Is this a dagger" scene — slug line, stage directions, soliloquy, sensory imagery):

| Stage | Wall time |
|---|---|
| Brain-response prediction | ~7 min 54 s |
| Behavioral analysis | ~4 min 45 s |
| **Total** | **~12.7 min** |

**Excerpt from the returned analysis** (full response was ~7 KB):

> **The Neural Take: This Opening Is a Trap — And the Brain Is Already Caught**
>
> **t=0–1s: The Blueprint Dump.** The instant the auditory stream hits "INT. MACBETH'S CASTLE," the brain executes a rapid schema download. Angular gyrus and temporal pole ignite — not because the words are loud, but because they're a *spatial password*. The listener pulls a pre-built medieval-fortress template from long-term memory and loads it into working space.
>
> **t=5–7s: Negative Space & Threat Detection.** Here's the inflection. "The corridor is empty." Emptiness in narrative is computationally vicious — the brain expects agents in a built environment. The explicit absence forces a controlled detonation of that social prediction, creating a *prediction error surge* in the TPJ. Then: "Distant, a bell…" An unexplained sound in a depopulated space. This is the Hitchcock effect made neural.
>
> **Peak activation: t≈6–7s** — collision of "The corridor is empty" with the bell's first toll. Cognitive mechanism: *liminal prediction error with perspective collapse*.
>
> **User behavior prediction:** zero off-ramp (no-scroll, no-skip stimulus), high episodic memory encoding, motor-planning regions primed for tremor (identity fusion with a fictional murderer), high social-transmission likelihood. *Bottom line: this stimulus isn't telling a story — it's executing a neural heist.*

---

## 6. Versioning / changelog

- **1.0.0** — initial REST surface (`POST /analysis`, `GET /healthz`).
- The legacy Gradio API (mounted at `/gradio_api/...`) and the Gradio UI (at `/`) remain available on the same host for the dashboard, but are not part of this contract.

---

## 7. Contact

Service owner: Neelanjan Mitra (`neelanjan.mitra@acti.ai`).
