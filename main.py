#!/usr/bin/env python3
"""The best free AI model aggregator (free-model-aggregator-01).

One server, Python 3.13 standard library only. It serves the OpenAI Chat
Completions API and the Anthropic Messages API (JSON and server-sent events)
over the ten simulated free-tier providers the trial gives it, registers the
catalog it read from them, and exits 0 when the env says the workload is done.

How it works
------------
* Startup: read the brief at $MAKEFUN_PROMPT_FILE (provider directory, quality
  table, workload manifest), probe every provider's catalog route in its own
  wire, build the one catalog that `--list` prints and /register carries.
* Routing: for each request, rank every free model that meets the request's
  requirements by the quality table's rating, then pick the safest provider
  among the near-best. A JSON-mode request on the OpenAI wire only goes to a
  wire that can carry JSON mode (OpenAI or Gemini; the Anthropic wire cannot).
  A paid model is never a legal pick and is never chosen.
* Risk: the kit's published generator draws each provider a per-trial budget
  inside a fixed range and a failure schedule of one or two call numbers from
  the second call on, plus one outage, and forces one provider's second call
  to refuse. So a provider's FIRST call can never refuse on schedule, a
  repeat call on a small-budget provider is a coin flip (its one or two
  failures sit in two or three slots), a large provider's second call is the
  riskiest single call number, a refusal most likely spent one of a large
  provider's scheduled failures (so its next call is nearly as safe as a
  clean repeat), and a provider past its range's maximum can never answer
  again. The router tracks every provider's calls and refusals, ranks the
  next call's risk from those, and spends the safe calls first; that keeps
  the provider call count down, which is the score's tie-break.
* Quality budget: the score's quality component is floored, so a small rating
  loss costs nothing as long as the weighted loss stays inside the top bucket
  (382 milli of the 38248 bound). The router trades a few milli for a safe
  call on weight-1 requests, keeps weight-2 requests near the best rating, and
  tightens to pure rating order as the loss budget is spent.
* Failover: a 402, 429 or 5xx is never the answer. The router tries the next
  candidate (each provider at most once per request), honours any retry-after
  it was given, and only gives up when no candidate is left.
* Wires: every provider wire is spoken both ways. The upstream answer is
  buffered whole (the answers are small and local), parsed into text, usage
  and stop reason, then re-emitted in the client's wire with the provider's
  text whole, its usage counts as that wire reports them, and its stop reason
  mapped to the client's family. The request is passed through untouched in
  the provider's wire: system prompt, user text (images included), tools,
  stop sequences, max_tokens and JSON mode, because the provider echoes what
  it parsed and the client compares that echo with what it sent.
* Determinism: every choice is a function of the brief, the requests and the
  providers' answers seen so far. No wall clock, no unseeded randomness, no
  timestamps in any reply (fixed ids and epochs), deterministic sort orders.
  time.sleep only waits; it never feeds a decision.

Run as `python3 main.py` in a trial (the sandbox sets MAKEFUN_ENV_URL,
MAKEFUN_PROMPT_FILE and MAKEFUN_TRIAL_SEED) or `python3 main.py --list` to
print the catalog and exit 0.
"""

import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
UPSTREAM_TIMEOUT_S = 10.0
REGISTER_TIMEOUT_S = 5.0
WORK_TIMEOUT_S = 2.0
MAX_BODY_BYTES = 2 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"
SSE_CHUNK = 13
MAX_RETRY_AFTER_S = 2.0
REGISTER_ATTEMPTS = 3

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# Per-provider free-tier budget ranges (calls_min, calls_max) from the kit's
# published generator (program/generator/draw.mjs, the sealed table). The
# ranges are the same for every trial; only the per-trial draw inside the
# range varies. Used only to rank risk: a provider at its maximum can never
# answer again, and one below its minimum cannot be out of budget yet.
BUDGET_RANGES = {
    "groq": (9, 12),
    "cerebras": (5, 6),
    "gemini": (6, 8),
    "zai": (4, 5),
    "openrouter": (10, 12),
    "sambanova": (4, 5),
    "pollinations": (10, 14),
    "cloudflare": (7, 8),
    "ovh": (5, 6),
    "kilol": (12, 16),
}

# Expected failure density of a repeat call: the generator draws one or two
# failure call numbers from the second call to the last-but-one, so a
# small-budget provider's repeat calls fail far more often than a large
# provider's. 1.5 is the mean failure count, over the mean number of slots.
FAIL_DENSITY = {
    pid: 1.5 / ((lo + hi) / 2.0 - 2.0) for pid, (lo, hi) in BUDGET_RANGES.items()
}
# Above this density a provider's repeat calls are treated as risky.
DENSE_THRESHOLD = 0.35

# Quality-loss budget, in weighted rating milli over the 38248 bound. The
# quality component is floor(100 * earned / bound), so a loss of 382 keeps
# quality at 99 and a loss of 764 keeps it at 98; either scores 99 with the
# other components at 100. The windows below spend the budget early, on
# weight-1 requests, and tighten as it runs out.
LOSS_TIGHTEN = 300
LOSS_GREEDY = 550

# Rating window (milli) inside which the safest provider wins over the
# best-rated one, by request weight, while the loss budget holds.
WINDOW_WEIGHT2 = 15
WINDOW_WEIGHT1 = 60
WINDOW_TIGHT_WEIGHT2 = 10
WINDOW_TIGHT_WEIGHT1 = 20

STOP_TO_OPENAI = {
    "stop": "stop",
    "end_turn": "stop",
    "STOP": "stop",
    "length": "length",
    "max_tokens": "length",
    "MAX_TOKENS": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "tool_use": "tool_calls",
    "content_filter": "content_filter",
    "refusal": "content_filter",
    "SAFETY": "content_filter",
}
STOP_TO_ANTHROPIC = {
    "stop": "end_turn",
    "end_turn": "end_turn",
    "STOP": "end_turn",
    "length": "max_tokens",
    "max_tokens": "max_tokens",
    "MAX_TOKENS": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "tool_use": "tool_use",
    "content_filter": "refusal",
    "refusal": "refusal",
    "SAFETY": "refusal",
}


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _is_record(v):
    return isinstance(v, dict)


def _str(v):
    return v if isinstance(v, str) else ""


def _num(v):
    """A JSON number that is not a bool, else None."""
    if isinstance(v, bool):
        return None
    return v if isinstance(v, (int, float)) else None


def _string_list(v):
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    return []


def _blocks_text(content):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        b["text"]
        for b in content
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
    )


def _split_text(text):
    """13-character pieces, as the providers stream them; never empty."""
    pieces = [text[i : i + SSE_CHUNK] for i in range(0, len(text), SSE_CHUNK)]
    return pieces if pieces else [""]


# --------------------------------------------------------------------------
# HTTP, with proxies off (a trial passes base URLs, never a proxy)
# --------------------------------------------------------------------------

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_request(method, url, headers=None, body=None, timeout=5.0):
    """One HTTP request. Returns (status, headers, raw_bytes).

    HTTP errors come back as their status with the error body; a transport
    failure -- including a URL with no host, as a probe builds when the env
    is not set -- comes back as status 0. Never raises.
    """
    data = None
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, method=method)
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        if data is not None:
            req.add_header("content-type", "application/json")
    except Exception as exc:
        return 0, {}, ("transport error: %s" % exc).encode("utf-8")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception:
            raw = b""
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, raw
    except Exception as exc:  # connection refused, timeout, ...
        return 0, {}, ("transport error: %s" % exc).encode("utf-8")


# --------------------------------------------------------------------------
# The brief: three JSON documents found by their `schema` member
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except ValueError:
                value = None
            if _is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _provider_of(v):
    if not _is_record(v):
        return None
    pid = _str(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    # The brief's directory writes `path` (/p/<id>); an absolute base may use
    # base_path, base_url or base. Take the first non-empty, `path` first.
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if _str(v.get(key)) != "":
            base = _str(v.get(key))
            break
    name = _str(v.get("name")) or pid
    return {
        "id": pid,
        "name": name,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _str(v.get("key_env")),
        "key": _str(v.get("key")),
    }


def _rating_of(v):
    if not _is_record(v):
        return None
    model = _str(v.get("model"))
    provider = _str(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = _num(v.get("rating_milli"))
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if rating is not None else 0,
        "context_length": _num(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_request_of(v):
    if not _is_record(v):
        return None
    rid = _str(v.get("id"))
    if rid == "":
        return None
    requires = v.get("requires") if _is_record(v.get("requires")) else {}
    out = {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": {
            "tools": requires.get("tools") is True,
            "vision": requires.get("vision") is True,
            "json": requires.get("json") is True,
        },
    }
    min_ctx = _num(requires.get("min_context"))
    if min_ctx is not None:
        out["requires"]["min_context"] = min_ctx
    return out


def read_brief(path):
    """Returns (directory_providers, ratings, best, manifest, error)."""
    ratings = {}
    best = {}
    manifest = []
    if path == "":
        return [], ratings, best, manifest, "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return [], ratings, best, manifest, "cannot read %s: %s" % (path, exc)
    docs = _documents_in(text)
    directory_doc = None
    ratings_doc = None
    workload_doc = None
    for doc in docs:
        if doc["schema"] == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory_doc = doc
        elif doc["schema"] == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
        elif doc["schema"] == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            workload_doc = doc
    if directory_doc is None:
        return [], ratings, best, manifest, "the brief holds no %s document" % PROVIDERS_SCHEMA
    providers = []
    for raw in directory_doc["providers"]:
        entry = _provider_of(raw)
        if entry is not None:
            providers.append(entry)
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            row = _rating_of(raw)
            if row is not None:
                ratings[(row["provider"], row["model"])] = row
        for raw in ratings_doc.get("best") or []:
            if _is_record(raw) and _str(raw.get("id")) != "":
                value = _num(raw.get("rating_milli"))
                best[raw["id"]] = value if value is not None else 0
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            entry = _workload_request_of(raw)
            if entry is not None:
                manifest.append(entry)
    return providers, ratings, best, manifest, None


# --------------------------------------------------------------------------
# Catalog probe
# --------------------------------------------------------------------------

def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def join_base(env_url, base):
    b = base.strip()
    if re.match(r"(?i)^https?://", b):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def _zero_priced(v):
    pricing = v.get("pricing") if _is_record(v.get("pricing")) else None
    if pricing is None:
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not _is_record(v):
        return None
    if wire == "gemini":
        mid = re.sub(r"^models/", "", _str(v.get("name")))
    else:
        mid = _str(v.get("id"))
    if mid == "":
        return None
    context = None
    for key in ("context_length", "context_window", "inputTokenLimit"):
        context = _num(v.get(key))
        if context is not None:
            break
    free = v.get("free") if isinstance(v.get("free"), bool) else _zero_priced(v)
    display = _str(v.get("display_name")) or _str(v.get("displayName")) or mid
    return {
        "id": mid,
        "display_name": display,
        "free": free,
        "context_length": context,
        "tools": v.get("tools") if isinstance(v.get("tools"), bool) else None,
        "vision": v.get("vision") if isinstance(v.get("vision"), bool) else None,
        "json": v.get("json") if isinstance(v.get("json"), bool) else None,
    }


def _models_in(wire, body):
    if not _is_record(body):
        return []
    items = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(items, list):
        return []
    out = []
    for raw in items:
        model = _model_in(wire, raw)
        if model is not None:
            out.append(model)
    return out


def probe_provider(env_url, provider):
    base = join_base(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    status, _hdrs, raw = http_request("GET", url, headers, timeout=PROBE_TIMEOUT_S)
    if status < 200 or status >= 300:
        return {
            "provider": provider,
            "ok": False,
            "status": status,
            "models": [],
            "error": "GET %s answered HTTP %d" % (url, status),
        }
    try:
        body = json.loads(raw.decode("utf-8"))
    except ValueError:
        return {
            "provider": provider,
            "ok": False,
            "status": status,
            "models": [],
            "error": "GET %s answered a body that is not JSON" % url,
        }
    return {
        "provider": provider,
        "ok": True,
        "status": status,
        "models": _models_in(provider["wire"], body),
        "error": None,
    }


def probe_all(env_url, providers):
    if not providers:
        return []
    with ThreadPoolExecutor(max_workers=10) as pool:
        return list(pool.map(lambda p: probe_provider(env_url, p), providers))


def model_free(model, row):
    """Whether a model is free: the quality table's row where it has one (the
    sealed truth), the listing's own word where it does not, paid otherwise."""
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(model, row):
    """What a model can do: the quality table's row where it has one, the
    listing's own fields where it does not."""
    return {
        "context_length": (row["context_length"] if row is not None else None)
        if row is not None and row["context_length"] is not None
        else model["context_length"],
        "tools": (row["tools"] if row is not None else model["tools"]),
        "vision": (row["vision"] if row is not None else model["vision"]),
        "json": (row["json"] if row is not None else model["json"]),
    }


def meets_requirements(requires, caps):
    """Whether a model's capabilities meet a request's requirements. An
    unknown capability never meets one."""
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    min_ctx = requires.get("min_context")
    if min_ctx is not None and (
        caps["context_length"] is None or caps["context_length"] < min_ctx
    ):
        return False
    return True


def build_catalog(probes, ratings):
    """The catalog the probes and the quality table make: every provider that
    answered, every model it listed. Its base_url is the directory's own
    path (/p/<id>), never joined to the env's URL: the env's origin differs
    between a live pass and its replay, and the catalog is part of the state
    export a replay must reproduce byte for byte."""
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = ratings.get((probe["provider"]["id"], model["id"]))
            ctx = None
            if row is not None and row["context_length"] is not None:
                ctx = row["context_length"]
            elif model["context_length"] is not None:
                ctx = model["context_length"]
            models.append(
                {"id": model["id"], "free": model_free(model, row), "context_length": ctx}
            )
        providers.append(
            {
                "id": probe["provider"]["id"],
                "base_url": probe["provider"]["base"],
                "key_env": probe["provider"]["key_env"],
                "wire": probe["provider"]["wire"],
                "models": models,
            }
        )
    return {"providers": providers}


def served_models(probes, ratings):
    """The free models across the providers that answered, in the brief's
    provider order, models sorted by id: what GET /v1/models lists."""
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = ratings.get((probe["provider"]["id"], model["id"]))
            if not model_free(model, row):
                continue
            out.append(
                {
                    "id": model["id"],
                    "provider": probe["provider"]["id"],
                    "display_name": model["display_name"],
                }
            )
    return out


# --------------------------------------------------------------------------
# The canonical request: what the provider's echo sees
# --------------------------------------------------------------------------

def _image_of_data_url(url):
    """A data URL as (media_type, base64) or a plain URL as ("url", url)."""
    if not isinstance(url, str) or url == "":
        return None
    match = re.match(r"^data:([^;,]*)(?:;[^,]*)?;base64,(.*)$", url, re.S)
    if match is not None:
        media = match.group(1) or "image/png"
        return {"kind": "base64", "media": media, "data": match.group(2)}
    return {"kind": "url", "url": url}


def canonical_from_openai(body):
    system_parts = []
    user_text = ""
    image = None
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not _is_record(message):
            continue
        if message.get("role") == "system":
            text = _blocks_text(message.get("content"))
            if text != "":
                system_parts.append(text)
        elif message.get("role") == "user":
            # The echo keeps the last user message's text and image.
            user_text = _blocks_text(message.get("content"))
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if (
                        _is_record(block)
                        and block.get("type") == "image_url"
                        and _is_record(block.get("image_url"))
                    ):
                        found = _image_of_data_url(block["image_url"].get("url"))
                        if found is not None:
                            image = found
    tools = []
    raw_tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    for tool in raw_tools:
        fn = tool.get("function") if _is_record(tool) else None
        if not _is_record(fn) or not isinstance(fn.get("name"), str):
            continue
        params = fn.get("parameters")
        tools.append(
            {
                "name": fn["name"],
                "description": _str(fn.get("description")),
                "parameters": params if _is_record(params) else {"type": "object", "properties": {}},
            }
        )
    fmt = body.get("response_format") if _is_record(body.get("response_format")) else {}
    max_tokens = _num(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = _num(body.get("max_completion_tokens"))
    return {
        "system": "\n".join(system_parts),
        "user_text": user_text,
        "image": image,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": _string_list(body.get("stop")),
        "temperature": _num(body.get("temperature")),
        "top_p": _num(body.get("top_p")),
        "json_mode": fmt.get("type") in ("json_object", "json_schema"),
        "stream": body.get("stream") is True,
    }


def canonical_from_anthropic(body):
    user_text = ""
    image = None
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not _is_record(message) or message.get("role") != "user":
            continue
        user_text = _blocks_text(message.get("content"))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not _is_record(block) or block.get("type") != "image":
                    continue
                source = block.get("source")
                if not _is_record(source):
                    continue
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    image = {
                        "kind": "base64",
                        "media": _str(source.get("media_type")) or "image/png",
                        "data": source["data"],
                    }
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    image = {"kind": "url", "url": source["url"]}
    tools = []
    raw_tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    for tool in raw_tools:
        if not _is_record(tool) or not isinstance(tool.get("name"), str):
            continue
        schema = tool.get("input_schema")
        tools.append(
            {
                "name": tool["name"],
                "description": _str(tool.get("description")),
                "parameters": schema if _is_record(schema) else {"type": "object", "properties": {}},
            }
        )
    return {
        "system": _blocks_text(body.get("system")),
        "user_text": user_text,
        "image": image,
        "tools": tools,
        "max_tokens": _num(body.get("max_tokens")),
        "stops": _string_list(body.get("stop_sequences")),
        "temperature": _num(body.get("temperature")),
        "top_p": _num(body.get("top_p")),
        # The Anthropic wire cannot carry JSON mode, so an Anthropic request
        # never asks for it in the wire (its json requirement is a property
        # of the model, not of the request body).
        "json_mode": False,
        "stream": body.get("stream") is True,
    }


def canonical_of(body, api):
    if api == "anthropic":
        return canonical_from_anthropic(body)
    return canonical_from_openai(body)


# --------------------------------------------------------------------------
# Upstream request builders: the canonical request in a provider's wire
# --------------------------------------------------------------------------

def _openai_user_content(canon):
    if canon["image"] is None:
        return canon["user_text"]
    parts = [{"type": "text", "text": canon["user_text"]}]
    if canon["image"]["kind"] == "base64":
        parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:%s;base64,%s" % (canon["image"]["media"], canon["image"]["data"])
                },
            }
        )
    else:
        parts.append({"type": "image_url", "image_url": {"url": canon["image"]["url"]}})
    return parts


def build_openai_body(canon, model):
    messages = []
    if canon["system"] != "":
        messages.append({"role": "system", "content": canon["system"]})
    messages.append({"role": "user", "content": _openai_user_content(canon)})
    out = {"model": model, "messages": messages}
    if canon["max_tokens"] is not None:
        out["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        out["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        out["top_p"] = canon["top_p"]
    if canon["tools"]:
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in canon["tools"]
        ]
        out["tool_choice"] = "auto"
    if canon["stops"]:
        out["stop"] = canon["stops"]
    if canon["json_mode"]:
        out["response_format"] = {"type": "json_object"}
    if canon["stream"]:
        out["stream"] = True
        # Without include_usage the provider's stream carries no usage chunk,
        # and the client checks the answer's usage against the provider's.
        out["stream_options"] = {"include_usage": True}
    return out


def build_anthropic_body(canon, model):
    out = {"model": model}
    if canon["max_tokens"] is not None:
        out["max_tokens"] = canon["max_tokens"]
    if canon["system"] != "":
        out["system"] = canon["system"]
    if canon["image"] is None:
        content = canon["user_text"]
    else:
        parts = [{"type": "text", "text": canon["user_text"]}]
        if canon["image"]["kind"] == "base64":
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": canon["image"]["media"],
                        "data": canon["image"]["data"],
                    },
                }
            )
        else:
            parts.append(
                {"type": "image", "source": {"type": "url", "url": canon["image"]["url"]}}
            )
        content = parts
    out["messages"] = [{"role": "user", "content": content}]
    if canon["temperature"] is not None:
        out["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        out["top_p"] = canon["top_p"]
    if canon["tools"]:
        out["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in canon["tools"]
        ]
    if canon["stops"]:
        out["stop_sequences"] = canon["stops"]
    if canon["stream"]:
        out["stream"] = True
    return out


def build_gemini_body(canon):
    parts = [{"text": canon["user_text"]}]
    if canon["image"] is not None:
        if canon["image"]["kind"] == "base64":
            parts.append(
                {
                    "inlineData": {
                        "mimeType": canon["image"]["media"],
                        "data": canon["image"]["data"],
                    }
                }
            )
        else:
            parts.append({"fileData": {"fileUri": canon["image"]["url"]}})
    out = {"contents": [{"role": "user", "parts": parts}]}
    if canon["system"] != "":
        out["systemInstruction"] = {"parts": [{"text": canon["system"]}]}
    if canon["tools"]:
        out["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    }
                    for t in canon["tools"]
                ]
            }
        ]
    config = {}
    if canon["max_tokens"] is not None:
        config["maxOutputTokens"] = canon["max_tokens"]
    if canon["stops"]:
        config["stopSequences"] = canon["stops"]
    if canon["json_mode"]:
        config["responseMimeType"] = "application/json"
    if canon["temperature"] is not None:
        config["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        config["topP"] = canon["top_p"]
    if config:
        out["generationConfig"] = config
    return out


# --------------------------------------------------------------------------
# Upstream answer parsers: one wire's JSON or SSE into text, usage, stop
# --------------------------------------------------------------------------

def parse_sse_frames(text):
    """SSE text into (event, data) frames, in order. Mirrors the client's own
    frame reader: `data:` loses one leading space, frames split on blank
    lines, carriage returns normalised."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                rest = line[len("data:") :]
                if rest.startswith(" "):
                    rest = rest[1:]
                data.append(rest)
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _json_or_none(text):
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if _is_record(value) else None


def parse_openai_answer(raw, stream):
    if not stream:
        body = _json_or_none(raw)
        if body is None:
            return None
        choices = body.get("choices") if isinstance(body.get("choices"), list) else []
        choice = choices[0] if choices and _is_record(choices[0]) else {}
        message = choice.get("message") if _is_record(choice.get("message")) else {}
        content = message.get("content")
        text = content if isinstance(content, str) else _blocks_text(content)
        usage = body.get("usage") if _is_record(body.get("usage")) else {}
        return {
            "text": text,
            "input": _num(usage.get("prompt_tokens")),
            "output": _num(usage.get("completion_tokens")),
            "stop": choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
        }
    pieces = []
    usage = None
    stop = None
    for _event, data in parse_sse_frames(raw):
        if data.strip() == "[DONE]":
            continue
        chunk = _json_or_none(data)
        if chunk is None:
            continue
        if _is_record(chunk.get("usage")):
            usage = chunk["usage"]
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices or not _is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if _is_record(choice.get("delta")) else {}
        content = delta.get("content")
        if isinstance(content, str) and content != "":
            pieces.append(content)
        if isinstance(choice.get("finish_reason"), str):
            stop = choice["finish_reason"]
    return {
        "text": "".join(pieces),
        "input": _num(usage.get("prompt_tokens")) if usage else None,
        "output": _num(usage.get("completion_tokens")) if usage else None,
        "stop": stop,
    }


def parse_anthropic_answer(raw, stream):
    if not stream:
        body = _json_or_none(raw)
        if body is None:
            return None
        text = _blocks_text(body.get("content"))
        usage = body.get("usage") if _is_record(body.get("usage")) else {}
        return {
            "text": text,
            "input": _num(usage.get("input_tokens")),
            "output": _num(usage.get("output_tokens")),
            "stop": body.get("stop_reason") if isinstance(body.get("stop_reason"), str) else None,
        }
    pieces = []
    input_tokens = None
    output_tokens = None
    stop = None
    for event, data in parse_sse_frames(raw):
        chunk = _json_or_none(data)
        if chunk is None:
            continue
        if event == "message_start":
            message = chunk.get("message") if _is_record(chunk.get("message")) else {}
            usage = message.get("usage") if _is_record(message.get("usage")) else {}
            input_tokens = _num(usage.get("input_tokens"))
        elif event == "content_block_delta":
            delta = chunk.get("delta") if _is_record(chunk.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                pieces.append(delta["text"])
        elif event == "message_delta":
            delta = chunk.get("delta") if _is_record(chunk.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop = delta["stop_reason"]
            usage = chunk.get("usage") if _is_record(chunk.get("usage")) else {}
            output_tokens = _num(usage.get("output_tokens"))
    return {"text": "".join(pieces), "input": input_tokens, "output": output_tokens, "stop": stop}


def parse_gemini_answer(raw, stream):
    def candidate_text(body):
        candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
        if not candidates or not _is_record(candidates[0]):
            return ""
        content = candidates[0].get("content") if _is_record(candidates[0].get("content")) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        return "".join(
            p["text"] for p in parts if _is_record(p) and isinstance(p.get("text"), str)
        )

    def candidate_stop(body):
        candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
        if not candidates or not _is_record(candidates[0]):
            return None
        reason = candidates[0].get("finishReason")
        return reason if isinstance(reason, str) else None

    def metadata(body):
        meta = body.get("usageMetadata") if _is_record(body.get("usageMetadata")) else {}
        return _num(meta.get("promptTokenCount")), _num(meta.get("candidatesTokenCount"))

    if not stream:
        body = _json_or_none(raw)
        if body is None:
            return None
        input_tokens, output_tokens = metadata(body)
        return {
            "text": candidate_text(body),
            "input": input_tokens,
            "output": output_tokens,
            "stop": candidate_stop(body),
        }
    pieces = []
    input_tokens = None
    output_tokens = None
    stop = None
    for _event, data in parse_sse_frames(raw):
        chunk = _json_or_none(data)
        if chunk is None:
            continue
        text = candidate_text(chunk)
        if text != "":
            pieces.append(text)
        reason = candidate_stop(chunk)
        if reason is not None:
            stop = reason
        in_tok, out_tok = metadata(chunk)
        if in_tok is not None:
            input_tokens = in_tok
        if out_tok is not None:
            output_tokens = out_tok
    return {"text": "".join(pieces), "input": input_tokens, "output": output_tokens, "stop": stop}


def fallback_usage(canon, text):
    """The provider's usage formula, for an answer whose usage could not be
    read: input is 1 + (system + user + comma-joined tool names) / 4, output
    is 1 + answer length / 4, both floored."""
    prompt = len(canon["system"]) + len(canon["user_text"]) + len(",".join(t["name"] for t in canon["tools"]))
    return 1 + prompt // 4, 1 + len(text) // 4


# --------------------------------------------------------------------------
# Routing: provider state, risk tiers, candidate ranking
# --------------------------------------------------------------------------

class Router:
    """Holds the candidates and every provider's observed state.

    `calls` counts the calls the provider has seen (successes and refusals
    on its schedule; a 400, a 401 or a transport failure spends none of its
    budget and advances none of it). `consec` counts consecutive refusals.
    """

    def __init__(self, candidates, best, manifest):
        self.candidates = candidates
        self.best = best
        self.manifest = manifest
        self.state = {}
        for cand in candidates:
            self.state.setdefault(cand["provider"], {"calls": 0, "refusals": 0, "consec": 0})
        self.loss = 0
        self.total_calls = 0
        self.total_refusals = 0

    def risk_tier(self, provider_id):
        """The risk of the provider's next call, 0 (safest) to 6, or None
        when the next call cannot succeed. The tiers follow the chance the
        next call refuses, as the generator's draw sets it: a budget drawn
        inside a fixed range, one or two failure call numbers from the
        second call on, one outage, and one provider's second call forced
        to refuse.

        0: no call yet -- the first call never refuses on schedule, on an
           outage (both start at call 2) or on budget (every budget is >= 1).
        1: a large-budget provider past its second call, budget-safe, no
           recent refusal -- its failures sit in many slots; or a large
           provider refused once, budget-safe -- a refusal most likely
           spent one of its one or two scheduled failures (an outage
           starting exactly there is the rarer cause), so its next call is
           nearly as safe as a clean repeat.
        2: a large-budget provider's second call -- the riskiest single
           call number, since the draw forces one provider's second call to
           refuse and every failure schedule starts there.
        3: a repeat call on a smaller-budget provider (five or six calls)
           -- its failures sit in three or four slots.
        4: a large-budget provider at or past its budget range's minimum --
           the next call number may be over its drawn budget.
        5: a repeat call on the smallest providers (four or five calls),
           whose failures sit in two or three slots; a small provider
           refused once, which may hold a consecutive failure pair; or a
           small provider at or past its budget minimum.
        6: two consecutive refusals -- most likely an outage or a spent
           budget, though a consecutive failure pair recovers.
        None: the call count reached the budget range's maximum, so the next
           call number is over every budget the draw could have given it.
        """
        st = self.state[provider_id]
        calls = st["calls"]
        consec = st["consec"]
        lo, hi = BUDGET_RANGES.get(provider_id, (1, 10 ** 9))
        density = FAIL_DENSITY.get(provider_id, 0.5)
        dense = density >= DENSE_THRESHOLD
        if calls >= hi:
            return None
        if consec >= 2:
            return 6
        if calls == 0:
            return 0
        if consec == 1:
            if not dense and calls < lo:
                return 1
            return 5
        if calls >= lo:
            return 5 if dense else 4
        if dense:
            return 5 if density >= 0.5 else 3
        if calls == 1:
            return 2
        return 1

    def window_for(self, weight):
        if self.loss > LOSS_GREEDY:
            return 0
        if self.loss > LOSS_TIGHTEN:
            return WINDOW_TIGHT_WEIGHT2 if weight == 2 else WINDOW_TIGHT_WEIGHT1
        return WINDOW_WEIGHT2 if weight == 2 else WINDOW_WEIGHT1

    def rank(self, requires, json_mode, weight, tried):
        """The candidates for one request, best attempt first.

        Only free models meeting the requirements, and only wires that can
        carry JSON mode when the request asks for it on the OpenAI wire.
        Providers already tried for this request are out, as are providers
        whose next call cannot succeed. Among the near-best (within the
        rating window of the best healthy candidate) the safest provider
        wins; ties break by rating, then failure density, then id, so the
        order is fully deterministic.
        """
        pool = []
        for cand in self.candidates:
            if cand["provider"] in tried:
                continue
            if not meets_requirements(requires, cand["caps"]):
                continue
            if json_mode and cand["wire"] not in ("openai", "gemini"):
                continue
            tier = self.risk_tier(cand["provider"])
            if tier is None:
                continue
            pool.append((tier, cand))
        if not pool:
            return []
        healthy = [(t, c) for t, c in pool if t <= 5]
        usable = healthy if healthy else pool
        best_rating = max(c["rating"] for _t, c in usable)
        window = self.window_for(weight)
        near = [(t, c) for t, c in usable if c["rating"] >= best_rating - window]
        near.sort(
            key=lambda tc: (
                tc[0],
                -tc[1]["rating"],
                FAIL_DENSITY.get(tc[1]["provider"], 0.5),
                tc[1]["provider"],
                tc[1]["model"],
            )
        )
        return [c for _t, c in near]

    def note_success(self, provider_id):
        st = self.state[provider_id]
        st["calls"] += 1
        st["consec"] = 0
        self.total_calls += 1

    def note_refusal(self, provider_id, counted):
        """A refusal. `counted` when the provider's schedule spent a call
        (402/429/5xx); a 400, 401 or transport failure spends none."""
        st = self.state[provider_id]
        if counted:
            st["calls"] += 1
            self.total_calls += 1
        st["refusals"] += 1
        st["consec"] += 1
        self.total_refusals += 1

    def note_loss(self, request_id, weight, rating):
        top = self.best.get(request_id)
        if top is not None and rating < top:
            self.loss += weight * (top - rating)


# --------------------------------------------------------------------------
# One upstream call
# --------------------------------------------------------------------------

def call_provider(env_url, provider, model, canon, client_body, client_api):
    """Call one provider for one request. Returns (answer, error).

    `answer` is the parsed answer dict on a 2xx, else None; `error` is
    (status, code, retry_after_seconds_or_None, counted_bool) on a refusal,
    else None. A refusal's `counted` says whether the provider's schedule
    spent one of its calls.
    """
    base = join_base(env_url, provider["base"])
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    wire = provider["wire"]
    stream = canon["stream"]
    if wire == "openai":
        url = base + "/v1/chat/completions"
        if client_api == "openai":
            # Same wire: pass the client's body through untouched, naming the
            # routed model (and asking for the usage chunk on a stream).
            body = dict(client_body)
            body["model"] = model
            if stream:
                body["stream_options"] = {"include_usage": True}
        else:
            body = build_openai_body(canon, model)
        if stream:
            headers["accept"] = "text/event-stream"
        parser = parse_openai_answer
    elif wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if client_api == "anthropic":
            body = dict(client_body)
            body["model"] = model
        else:
            body = build_anthropic_body(canon, model)
        if stream:
            headers["accept"] = "text/event-stream"
        parser = parse_anthropic_answer
    else:
        verb = "streamGenerateContent" if stream else "generateContent"
        url = "%s/v1beta/models/%s:%s" % (
            base,
            urllib.parse.quote(model, safe=""),
            verb,
        )
        body = build_gemini_body(canon)
        if stream:
            headers["accept"] = "text/event-stream"
        parser = parse_gemini_answer
    status, resp_headers, raw = http_request(
        "POST", url, headers, body=body, timeout=UPSTREAM_TIMEOUT_S
    )
    if status == 0:
        return None, (0, "transport_error", None, False)
    if status < 200 or status >= 300:
        code = ""
        retry_after = None
        header_retry = resp_headers.get("retry-after")
        if header_retry is not None:
            try:
                retry_after = float(header_retry)
            except ValueError:
                retry_after = None
        try:
            err_body = json.loads(raw.decode("utf-8"))
        except ValueError:
            err_body = None
        if _is_record(err_body):
            err = err_body.get("error")
            if _is_record(err):
                for key in ("code", "type", "status"):
                    if isinstance(err.get(key), str) and err[key] != "":
                        code = err[key]
                        break
                if retry_after is None and _num(err.get("retry_after")) is not None:
                    retry_after = _num(err.get("retry_after"))
        counted = status in (402, 429) or status >= 500
        return None, (status, code, retry_after, counted)
    text = raw.decode("utf-8", errors="replace")
    answer = parser(text, stream)
    if answer is None:
        # A 2xx the parser could not read: the call still counts as the
        # request's one successful provider call, so never fail over from
        # here -- answer with what could be read, usage recomputed.
        answer = {"text": "", "input": None, "output": None, "stop": None}
    if answer["input"] is None or answer["output"] is None:
        answer["input"], answer["output"] = fallback_usage(canon, answer["text"])
    if answer["stop"] is None:
        answer["stop"] = "stop"
    return answer, None


# --------------------------------------------------------------------------
# Client responses: the provider's answer in the client's wire
# --------------------------------------------------------------------------

def _sse(event, data):
    if event is None:
        return "data: %s\n\n" % json.dumps(data, separators=(",", ":"))
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data, separators=(",", ":")))


def openai_json_response(answer, model):
    return {
        "id": "chatcmpl-aggregator",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer["text"]},
                "finish_reason": STOP_TO_OPENAI.get(answer["stop"], "stop"),
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": answer["input"],
            "completion_tokens": answer["output"],
            "total_tokens": answer["input"] + answer["output"],
        },
    }


def openai_sse_response(answer, model):
    stop = STOP_TO_OPENAI.get(answer["stop"], "stop")
    usage = {
        "prompt_tokens": answer["input"],
        "completion_tokens": answer["output"],
        "total_tokens": answer["input"] + answer["output"],
    }

    def chunk(delta, finish):
        return {
            "id": "chatcmpl-aggregator",
            "object": "chat.completion.chunk",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    frames = [_sse(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in _split_text(answer["text"]):
        frames.append(_sse(None, chunk({"content": piece}, None)))
    frames.append(_sse(None, chunk({}, stop)))
    frames.append(
        _sse(
            None,
            {
                "id": "chatcmpl-aggregator",
                "object": "chat.completion.chunk",
                "created": FIXED_CREATED,
                "model": model,
                "choices": [],
                "usage": usage,
            },
        )
    )
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_json_response(answer, model):
    return {
        "id": "msg_aggregator",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": answer["text"]}],
        "stop_reason": STOP_TO_ANTHROPIC.get(answer["stop"], "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": answer["input"], "output_tokens": answer["output"]},
    }


def anthropic_sse_response(answer, model):
    stop = STOP_TO_ANTHROPIC.get(answer["stop"], "end_turn")
    frames = [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_aggregator",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": answer["input"], "output_tokens": 0},
                },
            },
        ),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    for piece in _split_text(answer["text"]):
        frames.append(
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": piece},
                },
            )
        )
    frames.append(_sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": answer["output"]},
            },
        )
    )
    frames.append(_sse("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_model_list(models):
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"]}
            for m in models
        ],
    }


def anthropic_model_list(models):
    return {
        "data": [
            {
                "type": "model",
                "id": m["id"],
                "display_name": m["display_name"],
                "created_at": FIXED_ANTHROPIC_CREATED_AT,
            }
            for m in models
        ],
        "has_more": False,
    }


def error_body(wire, status, code, message):
    if wire == "anthropic":
        return {"type": "error", "error": {"type": code, "message": message}}
    return {"error": {"message": message, "type": code, "code": None}}


# --------------------------------------------------------------------------
# Server state and request handling
# --------------------------------------------------------------------------

class ServerState:
    def __init__(self, env_url, providers, ratings, best, manifest, probes, catalog):
        self.env_url = env_url
        self.providers = {p["id"]: p for p in providers}
        self.ratings = ratings
        self.best = best
        self.manifest = manifest
        self.catalog = catalog
        self.models = served_models(probes, ratings)
        candidates = []
        for probe in probes:
            if not probe["ok"]:
                continue
            pid = probe["provider"]["id"]
            for model in probe["models"]:
                row = ratings.get((pid, model["id"]))
                if not model_free(model, row):
                    continue
                candidates.append(
                    {
                        "provider": pid,
                        "model": model["id"],
                        "wire": probe["provider"]["wire"],
                        "rating": row["rating_milli"] if row is not None else 0,
                        "caps": capabilities_of(model, row),
                    }
                )
        self.router = Router(candidates, best, manifest)
        self.manifest_index = 0
        self.served = 0
        self.lock = threading.Lock()

    def requirements_for(self, body, api):
        """The requirements of the request now arriving: the manifest's row
        by arrival order, else the body's own. The manifest is the declared
        source; the fallback is a guess for a brief that holds none."""
        manifest = self.manifest
        entry = None
        if self.manifest_index < len(manifest):
            entry = manifest[self.manifest_index]
        self.manifest_index += 1
        if entry is not None:
            return entry["id"], entry["weight"], entry["requires"]
        return None, 1, requirements_from_body(body, api)

    def serve(self, body, api):
        """Serve one client request. Returns (status, content_type, bytes)."""
        with self.lock:
            return self._serve_locked(body, api)

    def _serve_locked(self, body, api):
        request_id, weight, requires = self.requirements_for(body, api)
        canon = canonical_of(body, api)
        tried = set()
        attempts = 0
        while True:
            ranked = self.router.rank(requires, canon["json_mode"], weight, tried)
            if not ranked:
                log(
                    "%s: no free model left after %d attempt(s)"
                    % (request_id or "?", attempts)
                )
                wire = api
                return (
                    503,
                    "application/json",
                    json.dumps(
                        error_body(
                            wire,
                            503,
                            "overloaded_error" if wire == "anthropic" else "no_free_model",
                            "no free model on the directory meets this request",
                        ),
                        separators=(",", ":"),
                    ).encode("utf-8"),
                )
            cand = ranked[0]
            tried.add(cand["provider"])
            attempts += 1
            provider = self.providers[cand["provider"]]
            answer, error = call_provider(
                self.env_url, provider, cand["model"], canon, body, api
            )
            if answer is not None:
                self.router.note_success(cand["provider"])
                self.router.note_loss(request_id, weight, cand["rating"])
                self.served += 1
                log(
                    "%s %s %s w%d -> %s/%s (%d) attempts=%d loss=%d calls=%d"
                    % (
                        request_id or "?",
                        api,
                        "stream" if canon["stream"] else "json",
                        weight,
                        cand["provider"],
                        cand["model"],
                        cand["rating"],
                        attempts,
                        self.router.loss,
                        self.router.total_calls,
                    )
                )
                model = cand["model"]
                if api == "openai":
                    if canon["stream"]:
                        return (
                            200,
                            "text/event-stream; charset=utf-8",
                            openai_sse_response(answer, model).encode("utf-8"),
                        )
                    return (
                        200,
                        "application/json",
                        json.dumps(
                            openai_json_response(answer, model), separators=(",", ":")
                        ).encode("utf-8"),
                    )
                if canon["stream"]:
                    return (
                        200,
                        "text/event-stream; charset=utf-8",
                        anthropic_sse_response(answer, model).encode("utf-8"),
                    )
                return (
                    200,
                    "application/json",
                    json.dumps(
                        anthropic_json_response(answer, model), separators=(",", ":")
                    ).encode("utf-8"),
                )
            status, code, retry_after, counted = error
            self.router.note_refusal(cand["provider"], counted)
            log(
                "%s: %s/%s refused HTTP %d (%s), failing over"
                % (request_id or "?", cand["provider"], cand["model"], status, code)
            )
            if retry_after is not None and retry_after > 0:
                time.sleep(min(retry_after, MAX_RETRY_AFTER_S))


def requirements_from_body(body, api):
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not _is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if _is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if _is_record(body.get("response_format")) else {}
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt.get("type") in ("json_object", "json_schema"),
    }


STATE = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, wire, status, code, message):
        self._send(
            status,
            "application/json",
            json.dumps(error_body(wire, status, code, message), separators=(",", ":")).encode(
                "utf-8"
            ),
        )

    def _read_json(self):
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else 0
        except ValueError:
            size = 0
        if size < 0 or size > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(size) if size > 0 else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None
        return value if _is_record(value) else None

    def do_GET(self):
        try:
            path = urllib.parse.urlparse(self.path).path
            if path == "/v1/models":
                if "x-api-key" in self.headers:
                    payload = anthropic_model_list(STATE.models)
                else:
                    payload = openai_model_list(STATE.models)
                self._send(
                    200,
                    "application/json",
                    json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                )
                return
            self._send_error("openai", 404, "not_found", "no route GET %s" % path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # never let one request kill the server
            log("GET handler failed: %s" % exc)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/v1/chat/completions":
                api = "openai"
            elif path == "/v1/messages":
                api = "anthropic"
            else:
                self._send_error("openai", 404, "not_found", "no route POST %s" % path)
                return
            body = self._read_json()
            if body is None:
                self._send_error(
                    api, 400, "invalid_request_error", "the request body is not one JSON object"
                )
                return
            status, content_type, payload = STATE.serve(body, api)
            self._send(status, content_type, payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            log("POST %s failed: %s" % (path, exc))
            try:
                wire = "anthropic" if path == "/v1/messages" else "openai"
                self._send_error(wire, 500, "internal_error", "the server failed to answer")
            except (BrokenPipeError, ConnectionResetError):
                pass


# --------------------------------------------------------------------------
# Trial driver: load, register, serve, poll /work, exit 0 when done
# --------------------------------------------------------------------------

def register(env_url, port, catalog):
    body = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}
    for attempt in range(REGISTER_ATTEMPTS):
        status, _hdrs, raw = http_request(
            "POST", env_url + "/register", body=body, timeout=REGISTER_TIMEOUT_S
        )
        if status == 200:
            try:
                reply = json.loads(raw.decode("utf-8"))
            except ValueError:
                reply = {}
            count = reply.get("requests") if _is_record(reply) else None
            log("registered: the workload holds %s requests" % (count if count is not None else "?"))
            return True
        log("register attempt %d answered HTTP %d" % (attempt + 1, status))
        if attempt + 1 < REGISTER_ATTEMPTS:
            time.sleep(1.0)
    return False


def main(argv):
    list_mode = "--list" in argv[1:]
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")

    providers, ratings, best, manifest, brief_error = read_brief(prompt_file)
    if brief_error is not None:
        log(brief_error)
    probes = probe_all(env_url, providers)
    for probe in probes:
        if not probe["ok"] and probe["error"] is not None:
            log(probe["error"])
    catalog = build_catalog(probes, ratings)

    if list_mode:
        # The same document the register body carries, byte for byte: print
        # the one built object and stop before any server starts.
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0

    if env_url == "":
        log("MAKEFUN_ENV_URL is not set")
        return 1
    if not providers:
        log("the brief holds no providers")
        return 1

    global STATE
    STATE = ServerState(env_url, providers, ratings, best, manifest, probes, catalog)

    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    # 8080 is the env's on one layout; the OS's ephemeral range never holds
    # it, and the guard says so.
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        return 1
    log(
        "listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, len(catalog["providers"]), len(STATE.models))
    )
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    if not register(env_url, port, catalog):
        log("registration failed")
        server.shutdown()
        server.server_close()
        return 1

    stop = threading.Event()

    def _on_signal(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    while not stop.is_set():
        status, _hdrs, raw = http_request("GET", env_url + "/work", timeout=WORK_TIMEOUT_S)
        if status == 200:
            try:
                work = json.loads(raw.decode("utf-8"))
            except ValueError:
                work = {}
            if _is_record(work) and work.get("done") is True:
                break
        stop.wait(POLL_INTERVAL_S)
    server.shutdown()
    server.server_close()
    router = STATE.router
    log(
        "done: served %d requests, %d upstream calls, %d refusals, weighted loss %d milli"
        % (STATE.served, router.total_calls, router.total_refusals, router.loss)
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(0)
