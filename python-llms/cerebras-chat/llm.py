"""Cerebras Inference implementation of the Dataiku custom LLM contract.

DSS instantiates the BaseLLM subclass below once per connection, calls set_config()
with the values from llm.json, then process() / process_stream() per completion.
"""

import logging

# Provided by DSS at runtime and not installable from PyPI -- never add it to a
# requirements file.
from dataiku.llm.python import BaseLLM  # type: ignore[import-not-found]

from dku_cerebras import DEFAULT_BASE_URL
from dku_cerebras.client import CerebrasClient
from dku_cerebras.credentials import api_key_from_config
from dku_cerebras.mapping import build_payload, parse_completion, stream_to_dataiku_chunks
from dku_cerebras.models import VISION_MODELS

logger = logging.getLogger(__name__)


def _set_trace_attributes(span, attributes):
    """Record attributes on a DSS trace span, tolerating tracing API differences.

    Tracing is observability, never a reason to fail a completion.
    """
    if span is None:
        return
    try:
        for key, value in attributes.items():
            if value is not None:
                span.attributes[key] = value
    except Exception:
        logger.debug("Could not write trace attributes", exc_info=True)


class CerebrasLLM(BaseLLM):
    def __init__(self):
        self.config = {}
        self.plugin_config = {}
        self.model = None
        self.supports_images = False
        self.reasoning_effort = None
        self.include_reasoning_in_text = False
        self.extra_body = {}

    def set_config(self, config, plugin_config):
        self.config = config or {}
        self.plugin_config = plugin_config or {}

        # Presets hold only the credential -- shared or per-user, see credentials.py.
        # The endpoint lives on the connection so the two are managed independently.
        api_key = api_key_from_config(self.config)
        base_url = self.config.get("base_url") or DEFAULT_BASE_URL

        self.model = (self.config.get("model") or "").strip()
        if not self.model:
            raise ValueError(
                "No model configured. Set a Cerebras model ID such as gpt-oss-120b; see "
                "https://inference-docs.cerebras.ai/models/overview for the catalog."
            )

        # /v1/models does not report capabilities, so vision support comes from the
        # known list, with the checkbox as an override for anything newer.
        self.supports_images = (
            self.model in VISION_MODELS or bool(self.config.get("supports_images", False))
        )

        self.reasoning_effort = self.config.get("reasoning_effort") or None
        self.include_reasoning_in_text = bool(self.config.get("include_reasoning_in_text", False))
        self.extra_body = self.config.get("extra_body") or {}

        self.client = CerebrasClient(
            api_key=api_key,
            base_url=base_url,
            timeout=self.config.get("request_timeout", 300),
            max_retries=self.config.get("max_retries", 3),
        )

        logger.info("Cerebras connection configured for model %s at %s", self.model, base_url)

    def _build_payload(self, query, settings):
        return build_payload(
            model=self.model,
            query=query,
            settings=settings,
            supports_images=self.supports_images,
            reasoning_effort=self.reasoning_effort,
            extra_body=self.extra_body,
        )

    def process(self, query, settings, trace):
        payload = self._build_payload(query, settings)

        span = None
        try:
            span = trace.subspan("Cerebras chat completion")
        except Exception:
            logger.debug("Tracing unavailable", exc_info=True)
        _set_trace_attributes(span, {
            "model": self.model,
            "messageCount": len(payload.get("messages", [])),
            "stream": False,
        })

        response_data = self.client.chat_completion(payload)
        result = parse_completion(
            response_data,
            include_reasoning_in_text=self.include_reasoning_in_text,
        )

        # `_reasoning` is not part of the DSS response contract: surface it on the
        # trace for debugging, then drop it before returning.
        reasoning = result.pop("_reasoning", None)
        _set_trace_attributes(span, {
            "finishReason": result.get("finishReason"),
            "promptTokens": result.get("promptTokens"),
            "completionTokens": result.get("completionTokens"),
            "cerebrasTimeInfo": response_data.get("time_info"),
            "reasoning": reasoning or None,
        })

        return result

    def process_stream(self, query, settings, trace):
        payload = self._build_payload(query, settings)

        span = None
        try:
            span = trace.subspan("Cerebras chat completion (streaming)")
        except Exception:
            logger.debug("Tracing unavailable", exc_info=True)
        _set_trace_attributes(span, {
            "model": self.model,
            "messageCount": len(payload.get("messages", [])),
            "stream": True,
        })

        response = self.client.stream_chat_completion(payload)
        try:
            for chunk in stream_to_dataiku_chunks(
                response.iter_lines(),
                include_reasoning_in_text=self.include_reasoning_in_text,
            ):
                if "footer" in chunk:
                    _set_trace_attributes(span, chunk["footer"])
                yield chunk
        finally:
            response.close()
