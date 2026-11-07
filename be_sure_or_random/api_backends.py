"""API backend helpers for Fireworks and OpenRouter zero-shot runs.

OpenRouter models are pinned to a single endpoint tag ``provider/quant`` that
advertised logprobs support (preferring higher precision and usable context).
Fallbacks are disabled so routing cannot silently switch hosts. See:
https://www.lesswrong.com/posts/KsyoSAyBRXtwzSugg/not-pinning-your-openrouter-provider-might-invalidate-your
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Assistant content used to anchor the answer token. Fireworks honours this as a
# genuine continuation prefill (the first sampled token is the option itself). No
# OpenRouter provider does — see ``backend_uses_assistant_prefill``.
ANSWER_PREFIX = "ANSWER:"

# Model ID → pinned OpenRouter endpoint tag (provider_slug/quantization).
# Pins are live-verified to actually return usable top_logprobs. "Usable" means the
# per-token sequence reconstructs the message content (see
# ``experiments._extract_option_probabilities``); catalog claims and even a non-empty
# logprobs block are insufficient, because several providers return logprobs for the
# final token only.
OPENROUTER_LOGPROB_ENDPOINTS: Dict[str, str] = {}

# Models with no OpenRouter endpoint that returns usable per-token logprobs. Every
# endpoint advertising ``logprobs`` was probed live; the reason records what came back.
# Run these on the Fireworks backend instead (or not at all).
OPENROUTER_UNSUPPORTED_LOGPROB_MODELS: Dict[str, str] = {}


# Reasoning models that return an empty answer within max_tokens unless reasoning is
# disabled. Matches the Fireworks path (reasoning_effort="none") and the paper's
# no-chain-of-thought setting; verified by scripts/camera_ready_probe_openrouter.py.
OPENROUTER_REASONING_OFF_MODELS = set()


def backend_uses_assistant_prefill(backend: str) -> bool:
    """Whether to append an ``ANSWER:`` assistant turn to anchor the answer token.

    Fireworks honours it: the response content is just ``Y`` and the first logprob
    entry is the option token. No OpenRouter provider does — they either ignore it
    (identical output with and without) or, worse, treat it as a completed turn and
    derail entirely. So we omit it there and locate the answer token by parsing the
    model's own ``ANSWER:`` prefix out of the logprob sequence.
    """
    return resolve_api_backend(backend) == "fireworks"

# Backward-compatible alias used by older call sites / docs.
OPENROUTER_LOGPROB_PROVIDERS = OPENROUTER_LOGPROB_ENDPOINTS

# Optional aliases from Fireworks IDs → OpenRouter IDs when --api-backend openrouter
# is used with the legacy Fireworks-style names.
FIREWORKS_TO_OPENROUTER: Dict[str, str] = {}

OPENROUTER_MODELS = sorted(OPENROUTER_LOGPROB_ENDPOINTS.keys())
OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
OPENROUTER_GENERATION_URL = "https://openrouter.ai/api/v1/generation?id={generation_id}"
DEFAULT_API_CONCURRENCY = 32


def resolve_api_backend(explicit: Optional[str] = None) -> str:
    backend = (explicit or os.environ.get("API_BACKEND") or "fireworks").strip().lower()
    if backend not in {"fireworks", "openrouter"}:
        raise ValueError(f"Unsupported API_BACKEND={backend!r}; expected 'fireworks' or 'openrouter'.")
    return backend


def resolve_api_concurrency(explicit: Optional[int] = None) -> int:
    """Bounded parallel API requests. Default 32; set API_CONCURRENCY=1 for sequential."""
    if explicit is not None:
        value = int(explicit)
    else:
        raw = os.environ.get("API_CONCURRENCY")
        value = int(raw) if raw not in (None, "") else DEFAULT_API_CONCURRENCY
    if value < 1:
        raise ValueError(f"API concurrency must be >= 1, got {value}")
    return value


def resolve_openrouter_model(model_name: str) -> str:
    return FIREWORKS_TO_OPENROUTER.get(model_name, model_name)


def split_endpoint_tag(endpoint_tag: str) -> Tuple[str, Optional[str]]:
    """Split ``provider`` or ``provider/quant`` into (provider_slug, quantization)."""
    endpoint_tag = endpoint_tag.strip()
    if "/" not in endpoint_tag:
        return endpoint_tag, None
    provider, quant = endpoint_tag.split("/", 1)
    quant = quant.strip() or None
    return provider.strip(), quant


def resolve_openrouter_endpoint(model_name: str, explicit: Optional[str] = None) -> str:
    """Return pinned endpoint tag (provider or provider/quant)."""
    if explicit:
        return explicit.strip()
    env_provider = os.environ.get("OPENROUTER_PROVIDER")
    if env_provider:
        return env_provider.strip()
    resolved = resolve_openrouter_model(model_name)
    endpoint = OPENROUTER_LOGPROB_ENDPOINTS.get(resolved)
    if not endpoint:
        reason = OPENROUTER_UNSUPPORTED_LOGPROB_MODELS.get(resolved)
        if reason:
            raise ValueError(
                f"OpenRouter cannot serve usable logprobs for {resolved!r}: {reason}. "
                "Run this model on the Fireworks backend instead, or drop it from the sweep. "
                "Set OPENROUTER_PROVIDER to force an endpoint anyway (results will be rejected "
                "by the logprob validity check)."
            )
        raise ValueError(
            f"No pinned logprobs endpoint configured for OpenRouter model {resolved!r}. "
            f"Known models: {sorted(OPENROUTER_LOGPROB_ENDPOINTS)}. "
            "Set OPENROUTER_PROVIDER to override (e.g. 'parasail/fp8'), or pass api_provider=..."
        )
    return endpoint


def resolve_openrouter_provider(model_name: str, explicit: Optional[str] = None) -> str:
    """Backward-compatible alias: returns the full endpoint tag."""
    return resolve_openrouter_endpoint(model_name, explicit)


def fetch_openrouter_endpoints_snapshot(model_name: str) -> Dict[str, Any]:
    """Fetch live OpenRouter endpoints JSON for change-detection / provenance."""
    resolved = resolve_openrouter_model(model_name)
    url = OPENROUTER_ENDPOINTS_URL.format(model=resolved)
    req = urllib.request.Request(url, headers={"User-Agent": "be-sure-or-random"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "model": resolved,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
            "endpoints": [],
        }

    raw_endpoints = []
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    if isinstance(data, dict):
        raw_endpoints = data.get("endpoints") or []

    compact: List[Dict[str, Any]] = []
    for endpoint in raw_endpoints:
        params = endpoint.get("supported_parameters") or []
        compact.append(
            {
                "provider_name": endpoint.get("provider_name"),
                "name": endpoint.get("name"),
                "quantization": endpoint.get("quantization"),
                "status": endpoint.get("status"),
                "context_length": endpoint.get("context_length"),
                "max_completion_tokens": endpoint.get("max_completion_tokens"),
                "supports_logprobs": "logprobs" in params,
                "supported_parameters": sorted(params) if isinstance(params, list) else params,
            }
        )

    return {
        "model": resolved,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_url": url,
        "n_endpoints": len(compact),
        "endpoints": compact,
    }


def get_api_client(backend: Optional[str] = None):
    from dotenv import load_dotenv
    from openai import OpenAI

    backend = resolve_api_backend(backend)
    # Prefer repo-root .env so keys resolve even when find_dotenv() can't walk frames.
    repo_env = Path(__file__).resolve().parents[1] / ".env"
    if repo_env.exists():
        load_dotenv(repo_env)
    else:
        load_dotenv()
    if backend == "fireworks":
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY environment variable is not set.")
        return OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    headers: Dict[str, str] = {
        # Opt into per-response routing provenance (served provider, attempts, etc.).
        "X-OpenRouter-Metadata": "enabled",
    }
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_X_TITLE", "be-sure-or-random")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers=headers,
    )


def build_chat_completion_kwargs(
    *,
    model_name: str,
    messages: list,
    backend: Optional[str] = None,
    provider: Optional[str] = None,
    top_logprobs: Optional[int] = None,
) -> Dict[str, Any]:
    """Build kwargs for client.chat.completions.create."""
    backend = resolve_api_backend(backend)
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_logprobs if top_logprobs is not None else (5 if backend == "openrouter" else 2),
    }
    if backend == "fireworks":
        # Fireworks reasoning models: disable chain-of-thought for next-token scoring.
        kwargs["reasoning_effort"] = "none"
        return kwargs

    # OpenRouter: resolve model aliases + hard-pin endpoint (provider/quant) with no fallbacks.
    resolved_model = resolve_openrouter_model(model_name)
    endpoint_tag = resolve_openrouter_endpoint(model_name, provider)
    provider_slug, quantization = split_endpoint_tag(endpoint_tag)
    # Prefer full endpoint tag when quantization is known; else provider slug only.
    pin = endpoint_tag if quantization else provider_slug
    provider_prefs: Dict[str, Any] = {
        "order": [pin],
        "only": [pin],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    if quantization:
        # Belt-and-suspenders if the router accepts either form.
        provider_prefs["quantizations"] = [quantization]

    kwargs["model"] = resolved_model
    kwargs["extra_body"] = {"provider": provider_prefs}
    if resolved_model in OPENROUTER_REASONING_OFF_MODELS:
        kwargs["extra_body"]["reasoning"] = {"enabled": False}
    return kwargs


def _selected_provider_from_metadata(metadata: Any) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None
    endpoints = metadata.get("endpoints")
    if isinstance(endpoints, dict):
        for key in ("available", "candidates", "endpoints", "selected"):
            entries = endpoints.get(key)
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and entry.get("selected"):
                        return entry.get("provider") or entry.get("name")
                for entry in entries:
                    if isinstance(entry, dict) and entry.get("provider"):
                        # Fall back to first listed if none marked selected.
                        if key == "selected":
                            return entry.get("provider")
        # Some payloads nest a single selected object.
        selected = endpoints.get("selected")
        if isinstance(selected, dict):
            return selected.get("provider") or selected.get("name")
    attempts = metadata.get("attempts")
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if isinstance(attempt, dict) and attempt.get("status") in (200, "200", None):
                if attempt.get("provider"):
                    return attempt.get("provider")
        for attempt in reversed(attempts):
            if isinstance(attempt, dict) and attempt.get("provider"):
                return attempt.get("provider")
    summary = metadata.get("summary")
    if isinstance(summary, str) and "selected=" in summary:
        return summary.split("selected=", 1)[1].split(",", 1)[0].strip() or None
    return None


def extract_openrouter_provenance(response: Any, response_headers: Optional[Any] = None) -> Dict[str, Any]:
    """Extract per-response routing provenance for research logging."""
    generation_id = getattr(response, "id", None)
    response_model = getattr(response, "model", None)
    system_fingerprint = getattr(response, "system_fingerprint", None)

    metadata = None
    extra = getattr(response, "model_extra", None)
    if isinstance(extra, dict):
        metadata = extra.get("openrouter_metadata")
    if metadata is None and hasattr(response, "openrouter_metadata"):
        metadata = getattr(response, "openrouter_metadata")
    if metadata is None and hasattr(response, "model_dump"):
        try:
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                metadata = dumped.get("openrouter_metadata")
                generation_id = generation_id or dumped.get("id")
                response_model = response_model or dumped.get("model")
                system_fingerprint = system_fingerprint or dumped.get("system_fingerprint")
        except Exception:
            pass

    header_generation_id = None
    header_provider = None
    if response_headers is not None:
        try:
            header_generation_id = response_headers.get("x-generation-id") or response_headers.get("X-Generation-Id")
            header_provider = response_headers.get("x-provider") or response_headers.get("X-Provider")
        except Exception:
            pass

    served_provider = _selected_provider_from_metadata(metadata) or header_provider
    compact_meta = None
    if isinstance(metadata, dict):
        compact_meta = {
            "strategy": metadata.get("strategy"),
            "summary": metadata.get("summary"),
            "attempt": metadata.get("attempt"),
            "region": metadata.get("region"),
            "requested": metadata.get("requested"),
            "is_byok": metadata.get("is_byok"),
            "attempts": metadata.get("attempts"),
        }

    return {
        "generation_id": generation_id or header_generation_id,
        "response_model": response_model,
        "system_fingerprint": system_fingerprint,
        "served_provider": served_provider,
        "openrouter_metadata": compact_meta,
    }


def create_chat_completion_with_provenance(
    client: Any,
    create_kwargs: Dict[str, Any],
    *,
    backend: str,
):
    """Create a chat completion; for OpenRouter, also return routing provenance."""
    if backend != "openrouter":
        return client.chat.completions.create(**create_kwargs), {}

    # Prefer raw response so we can read generation-id headers if body metadata is missing.
    if hasattr(client.chat.completions, "with_raw_response"):
        raw = client.chat.completions.with_raw_response.create(**create_kwargs)
        response = raw.parse()
        headers = getattr(raw, "headers", None)
        return response, extract_openrouter_provenance(response, headers)

    response = client.chat.completions.create(**create_kwargs)
    return response, extract_openrouter_provenance(response)
