"""Translation between the Dataiku LLM Mesh contract and the Cerebras chat completions API.

Everything here is pure: no network, no DSS imports. That keeps the tricky parts
(message rewriting, streaming assembly) testable outside DSS -- see tests/.

Cerebras is OpenAI-compatible, so the shapes below are the OpenAI ones with a few
Cerebras specifics called out inline.
"""

import json
import logging
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Dataiku CompletionSettings key -> Cerebras payload key, for settings that map 1:1.
SIMPLE_SETTINGS = {
    "temperature": "temperature",
    "topP": "top_p",
    "maxOutputTokens": "max_completion_tokens",
    "presencePenalty": "presence_penalty",
    "frequencyPenalty": "frequency_penalty",
    "stopSequences": "stop",
    "logitBias": "logit_bias",
    "seed": "seed",
    "logProbs": "logprobs",
    "topLogProbs": "top_logprobs",
}

# Cerebras accepts PNG and JPEG only, and only as base64 data URIs:
# https://inference-docs.cerebras.ai/capabilities/image-inputs
SUPPORTED_IMAGE_MIME_TYPES = frozenset(["image/png", "image/jpeg"])

IMAGE_INPUTS_DOC_URL = "https://inference-docs.cerebras.ai/capabilities/image-inputs"

# Base64 prefixes ("magic numbers") used to label inline images with a MIME type.
# Detecting rather than assuming means an unsupported format is named in the error
# the user sees, instead of going out as a mislabelled payload and coming back a 400.
# GIF and WebP are listed precisely because they are *not* supported -- recognising
# them is what lets the rejection say which format was supplied.
_BASE64_SIGNATURES = (
    ("iVBORw0KGgo", "image/png"),
    ("/9j/", "image/jpeg"),
    ("R0lGODlh", "image/gif"),
    ("UklGR", "image/webp"),
)

_DATA_URI_SCHEME = "data:"

STREAM_DONE_SENTINEL = "[DONE]"


class UnsupportedPromptError(ValueError):
    """Raised when a prompt asks for something this model cannot do."""


def detect_mime_type(b64_string):
    """MIME type for a base64 payload from its magic number, or None if unrecognised.

    Returns None rather than guessing: an unrecognised payload is rejected, not
    relabelled as PNG and sent anyway.
    """
    for signature, mime_type in _BASE64_SIGNATURES:
        if b64_string.startswith(signature):
            return mime_type
    return None


def data_uri_mime_type(url):
    """The MIME type declared by a `data:<type>;base64,...` URI, lowercased.

    Returns "" when the URI declares none, as in `data:;base64,...`, which is then
    rejected like any other unsupported format.
    """
    header = url[len(_DATA_URI_SCHEME):].split(",", 1)[0]
    return header.split(";")[0].strip().lower()


def _unsupported_format_error(mime_type):
    return UnsupportedPromptError(
        "Cerebras accepts PNG and JPEG images only, but this prompt carries {}. "
        "Convert the image to PNG or JPEG and try again. See {}".format(
            mime_type or "an unrecognised image format", IMAGE_INPUTS_DOC_URL
        )
    )


def _safe_url_summary(url):
    """`scheme://host/...` for an error message, dropping the path and query string.

    Those can carry signed tokens, and this string reaches the DSS logs.
    """
    split = urlsplit(url)
    if split.scheme and split.netloc:
        return "{}://{}/...".format(split.scheme, split.netloc)
    return url[:40]


def normalize_tool_calls(tool_calls):
    """Coerce assistant tool calls into the OpenAI wire format.

    Accepts both the nested `{"function": {...}}` shape and the flatter variants
    DSS has used, and always emits `arguments` as a JSON string.
    """
    normalized = []
    for index, tool_call in enumerate(tool_calls or []):
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name") or tool_call.get("functionName")
        arguments = function.get("arguments")
        if arguments is None:
            arguments = tool_call.get("arguments")
        if arguments is None:
            arguments = "{}"
        elif not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        call_id = tool_call.get("id") or tool_call.get("callId") or "call_{}".format(index)
        normalized.append({
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        })
    return normalized


def _build_content_parts(message, supports_images):
    """Turn a Dataiku multimodal `parts` list into an OpenAI `content` array."""
    role = message.get("role")
    content_parts = []

    for part in message.get("parts", []):
        part_type = part.get("type")

        if part_type == "TEXT":
            content_parts.append({"type": "text", "text": part.get("text", "")})
            continue

        if part_type not in ("IMAGE_INLINE", "IMAGE_URL", "IMAGE_URI"):
            logger.warning("Ignoring unsupported prompt part of type %s", part_type)
            continue

        if not supports_images:
            raise UnsupportedPromptError(
                "The prompt contains an image but this connection is not configured for a "
                "vision-capable model. Set Model to one that accepts image inputs (e.g. "
                "qwen-3.8-27b), or tick 'Force image input support' on the connection if "
                "it is a vision model the plugin does not recognise yet."
            )
        if role != "user":
            # Image parts are only accepted on user messages.
            raise UnsupportedPromptError(
                "Cerebras only accepts image inputs on user messages, got role '{}'.".format(role)
            )

        if part_type == "IMAGE_INLINE":
            inline_image = part.get("inlineImage", "")
            mime_type = detect_mime_type(inline_image)
            if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
                raise _unsupported_format_error(mime_type)
            url = "data:{};base64,{}".format(mime_type, inline_image)
        else:
            source_url = part.get("url") or part.get("uri") or ""
            if not source_url.startswith(_DATA_URI_SCHEME):
                # The API does not fetch remote images, and neither does this plugin:
                # downloading a URL chosen by whoever wrote the prompt would make the
                # DSS backend fetch arbitrary addresses on their behalf, including ones
                # only it can reach. The image has to arrive in the prompt itself.
                raise UnsupportedPromptError(
                    "Cerebras does not accept remote image URLs, and this plugin does not "
                    "download them. Embed the image in the prompt as a base64 PNG or JPEG "
                    "data URI instead of linking to {}. See {}".format(
                        _safe_url_summary(source_url) or "a remote URL", IMAGE_INPUTS_DOC_URL
                    )
                )
            mime_type = data_uri_mime_type(source_url)
            if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
                raise _unsupported_format_error(mime_type)
            url = source_url

        content_parts.append({"type": "image_url", "image_url": {"url": url}})

    return content_parts


def build_messages(messages, supports_images=False):
    """Rewrite a Dataiku `query["messages"]` list into Cerebras `messages`."""
    built = []

    for message in messages or []:
        role = message.get("role")

        if role == "assistant" and message.get("toolCalls"):
            built.append({
                "role": "assistant",
                # Must be present alongside tool_calls, even when null.
                "content": message.get("content"),
                "tool_calls": normalize_tool_calls(message["toolCalls"]),
            })

        elif role == "tool" and message.get("toolOutputs"):
            # DSS batches every tool result into one message; the API wants one per call.
            for tool_output in message["toolOutputs"]:
                output = tool_output.get("output", "")
                built.append({
                    "role": "tool",
                    "tool_call_id": tool_output.get("callId") or tool_output.get("id"),
                    "content": output if isinstance(output, str) else json.dumps(output),
                })

        elif "parts" in message:
            built.append({
                "role": role,
                "content": _build_content_parts(message, supports_images),
            })

        else:
            built.append({"role": role, "content": message.get("content", "")})

    return built


def _build_tool_choice(tool_choice):
    """Map a Dataiku toolChoice to the Cerebras `tool_choice` field."""
    if not tool_choice:
        return None

    if isinstance(tool_choice, str):
        value = tool_choice.lower()
        return value if value in ("auto", "none", "required") else None

    choice_type = (tool_choice.get("type") or "").lower()
    if choice_type in ("auto", "none", "required"):
        return choice_type
    if choice_type in ("function", "tool"):
        name = tool_choice.get("name") or (tool_choice.get("function") or {}).get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return None


# Where nested schemas sit, keyed by how they are held: one schema, a list of them,
# or a name -> schema mapping. Walking these positions explicitly rather than every
# dict avoids mistaking a property *named* "properties" for a schema.
_NESTED_SCHEMA = ("items", "not", "if", "then", "else")
_NESTED_SCHEMA_LIST = ("anyOf", "allOf", "oneOf", "prefixItems")
_NESTED_SCHEMA_MAP = ("properties", "$defs", "definitions", "patternProperties")


def _prepare_strict_schema(schema):
    """Copy `schema` with additionalProperties:false on every object.

    Strict decoding rejects the request unless every object carries the key, and
    DSS routinely omits it -- so it is injected rather than demanded.

    Returns (prepared, strictable). `strictable` is False when the schema asks for
    additional properties somewhere, which strict mode cannot honour; the caller
    then sends the original schema unstrict instead of silently overruling it.
    """
    if not isinstance(schema, dict):
        return schema, True

    prepared = dict(schema)
    strictable = True

    for key in _NESTED_SCHEMA:
        if isinstance(prepared.get(key), dict):
            prepared[key], ok = _prepare_strict_schema(prepared[key])
            strictable = strictable and ok

    for key in _NESTED_SCHEMA_LIST:
        if isinstance(prepared.get(key), list):
            children = [_prepare_strict_schema(child) for child in prepared[key]]
            prepared[key] = [child for child, _ in children]
            strictable = strictable and all(ok for _, ok in children)

    for key in _NESTED_SCHEMA_MAP:
        if isinstance(prepared.get(key), dict):
            walked = {}
            for name, child in prepared[key].items():
                walked[name], ok = _prepare_strict_schema(child)
                strictable = strictable and ok
            prepared[key] = walked

    if prepared.get("type") == "object" or "properties" in prepared:
        allows_extra = prepared.get("additionalProperties")
        if allows_extra is None:
            prepared["additionalProperties"] = False
        elif allows_extra is not False:
            strictable = False

    return prepared, strictable


def _build_response_format(response_format):
    """Map a Dataiku responseFormat to the Cerebras `response_format` field."""
    if not response_format:
        return None

    format_type = (response_format.get("type") or "").lower()
    schema = response_format.get("schema") or response_format.get("jsonSchema")

    if schema:
        prepared, strictable = _prepare_strict_schema(schema)
        json_schema = {
            "name": response_format.get("name") or "response",
            "schema": prepared if strictable else schema,
        }
        if strictable:
            json_schema["strict"] = True
        else:
            # Unstrict still honours the schema as a prompt-level contract; the model
            # just is not constrained to it during decoding.
            logger.warning(
                "Not requesting strict decoding: the schema allows additional "
                "properties, which strict mode cannot express."
            )
        return {"type": "json_schema", "json_schema": json_schema}
    if format_type in ("json", "json_object"):
        return {"type": "json_object"}
    return None


def coerce_extra_body(extra_body):
    """Parse a DSS MAP param into payload fields.

    DSS MAP values are always strings, so `"true"` and `"{\"a\": 1}"` are decoded
    into real JSON types; anything that is not valid JSON is passed through as text.
    """
    coerced = {}
    for key, raw_value in (extra_body or {}).items():
        if not isinstance(raw_value, str):
            coerced[key] = raw_value
            continue
        try:
            coerced[key] = json.loads(raw_value)
        except (TypeError, ValueError):
            coerced[key] = raw_value
    return coerced


def build_payload(model, query, settings, supports_images=False, reasoning_effort=None,
                  extra_body=None):
    """Build the JSON body for POST /v1/chat/completions.

    The `stream` field is deliberately absent: CerebrasClient sets it, so it cannot
    disagree with the HTTP read mode.
    """
    settings = settings or {}

    payload = {
        "model": model,
        "messages": build_messages(query.get("messages"), supports_images),
    }

    for dss_key, cerebras_key in SIMPLE_SETTINGS.items():
        if settings.get(dss_key) is not None:
            payload[cerebras_key] = settings[dss_key]

    tools = settings.get("tools")
    if tools:
        payload["tools"] = tools
        tool_choice = _build_tool_choice(settings.get("toolChoice"))
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    response_format = _build_response_format(settings.get("responseFormat"))
    if response_format:
        if tools:
            # Some models reject requests carrying both, and the ones that accept them
            # prioritise tool calling, so response_format holds either way.
            logger.warning(
                "Dropping response_format: Cerebras does not support combining it with tools."
            )
        else:
            payload["response_format"] = response_format

    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    payload.update(coerce_extra_body(extra_body))
    return payload


def _usage_to_dataiku(usage):
    """Map Cerebras usage onto the DSS token counters."""
    usage = usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        # Some responses report only the total.
        completion_tokens = max((usage.get("total_tokens", 0) or 0) - prompt_tokens, 0)
    return {"promptTokens": prompt_tokens, "completionTokens": completion_tokens}


def _wrap_reasoning(reasoning):
    return "<thinking>\n{}\n</thinking>\n\n".format(reasoning.strip())


def parse_completion(response_data, include_reasoning_in_text=False):
    """Turn a Cerebras chat completion into a Dataiku CompletionResponse.

    Returns the text, token counts, tool calls and finish reason, plus a
    non-contractual `_reasoning` key the caller can push onto the trace.
    """
    result = {
        "text": "",
        "promptTokens": 0,
        "completionTokens": 0,
        "toolCalls": [],
        "finishReason": "stop",
    }

    choices = response_data.get("choices") or []
    if choices:
        choice = choices[0]
        message = choice.get("message") or {}

        reasoning = message.get("reasoning") or ""
        text = message.get("content") or ""
        if reasoning and include_reasoning_in_text:
            text = _wrap_reasoning(reasoning) + text
        result["text"] = text
        result["_reasoning"] = reasoning

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            result["toolCalls"] = normalize_tool_calls(tool_calls)
            # The finish reason must say tool_calls or DSS will not run the tools.
            result["finishReason"] = "tool_calls"
        else:
            result["finishReason"] = choice.get("finish_reason") or "stop"

    result.update(_usage_to_dataiku(response_data.get("usage")))
    return result


def parse_stream_line(line):
    """Parse one SSE line into a dict. Returns None for keep-alives and comments."""
    if not line or not line.startswith("data:"):
        return None

    payload = line[len("data:"):].strip()
    if not payload:
        return None
    if payload == STREAM_DONE_SENTINEL:
        return {"_done": True}

    try:
        return json.loads(payload)
    except ValueError:
        logger.warning("Ignoring unparseable SSE chunk: %r", payload[:200])
        return None


def _merge_tool_call_deltas(accumulator, deltas):
    """Accumulate partial tool calls; `arguments` arrives split across chunks."""
    for delta in deltas:
        index = delta.get("index", 0)
        slot = accumulator.setdefault(
            index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if delta.get("id"):
            slot["id"] = delta["id"]
        function = delta.get("function") or {}
        if function.get("name"):
            slot["function"]["name"] = function["name"]
        if function.get("arguments"):
            slot["function"]["arguments"] += function["arguments"]


def stream_to_dataiku_chunks(lines, include_reasoning_in_text=False):
    """Convert an SSE line iterator into the chunk dicts DSS expects.

    Yields `{"text": ...}` as content arrives, then one `{"toolCalls": [...]}` once
    every call is fully assembled, then a final `{"footer": {...}}` carrying the
    token counts and finish reason.
    """
    tool_calls = {}
    usage = None
    finish_reason = None
    reasoning_open = False

    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        chunk = parse_stream_line(line)
        if chunk is None:
            continue
        if chunk.get("_done"):
            break

        # Usage arrives on the terminal chunk, which may carry no choices.
        if chunk.get("usage"):
            usage = chunk["usage"]

        # Type-guarded: a malformed chunk would otherwise raise mid-stream.
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            continue

        for choice in choices:
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning")
            if reasoning and include_reasoning_in_text:
                if not reasoning_open:
                    yield {"text": "<thinking>\n"}
                    reasoning_open = True
                yield {"text": reasoning}

            content = delta.get("content")
            if content:
                if reasoning_open:
                    yield {"text": "\n</thinking>\n\n"}
                    reasoning_open = False
                yield {"text": content}

            if delta.get("tool_calls"):
                _merge_tool_call_deltas(tool_calls, delta["tool_calls"])

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    if reasoning_open:
        yield {"text": "\n</thinking>\n\n"}

    assembled_tool_calls = [tool_calls[index] for index in sorted(tool_calls)]
    if assembled_tool_calls:
        yield {"toolCalls": normalize_tool_calls(assembled_tool_calls)}

    footer = _usage_to_dataiku(usage)
    footer["finishReason"] = "tool_calls" if assembled_tool_calls else (finish_reason or "stop")
    yield {"footer": footer}
