"""Unit tests for the Dataiku <-> Cerebras translation layer.

These run outside DSS (no dataiku import), so `pytest tests/` is enough:

    python -m pytest tests/ -q
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-lib"))

from dku_cerebras.mapping import (  # noqa: E402
    UnsupportedPromptError,
    build_messages,
    build_payload,
    coerce_extra_body,
    data_uri_mime_type,
    detect_mime_type,
    normalize_tool_calls,
    parse_completion,
    parse_stream_line,
    stream_to_dataiku_chunks,
)


def _sse(obj):
    return "data: " + json.dumps(obj)


class TestBuildMessages:
    def test_plain_messages_pass_through(self):
        messages = build_messages([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ])
        assert messages == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ]

    def test_assistant_tool_calls_are_renamed_and_arguments_stringified(self):
        messages = build_messages([{
            "role": "assistant",
            "toolCalls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Sunnyvale"}},
            }],
        }])
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"city": "Sunnyvale"}'
        assert "toolCalls" not in messages[0]

    def test_tool_outputs_expand_to_one_message_per_call(self):
        messages = build_messages([{
            "role": "tool",
            "toolOutputs": [
                {"callId": "call_1", "output": "21C"},
                {"callId": "call_2", "output": {"aqi": 30}},
            ],
        }])
        assert [m["tool_call_id"] for m in messages] == ["call_1", "call_2"]
        assert messages[0]["content"] == "21C"
        assert messages[1]["content"] == '{"aqi": 30}'

    def test_inline_image_becomes_a_data_uri(self):
        messages = build_messages(
            [{"role": "user", "parts": [
                {"type": "TEXT", "text": "What is this?"},
                {"type": "IMAGE_INLINE", "inlineImage": "iVBORw0KGgoAAAA"},
            ]}],
            supports_images=True,
        )
        content = messages[0]["content"]
        assert content[0] == {"type": "text", "text": "What is this?"}
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,iVBORw0KGgo")

    def test_a_jpeg_data_uri_is_passed_through_unchanged(self):
        source = "data:image/jpeg;base64,/9j/4AAQ"
        messages = build_messages(
            [{"role": "user", "parts": [{"type": "IMAGE_URL", "url": source}]}],
            supports_images=True,
        )
        assert messages[0]["content"][0]["image_url"]["url"] == source

    def test_a_remote_image_url_is_rejected_rather_than_downloaded(self):
        # The plugin must never fetch a prompt-supplied URL on the backend's behalf.
        with pytest.raises(UnsupportedPromptError, match="does not accept remote image URLs"):
            build_messages(
                [{"role": "user", "parts": [
                    {"type": "IMAGE_URL", "url": "https://x.example/y.png"},
                ]}],
                supports_images=True,
            )

    def test_the_rejection_names_the_host_but_not_the_path_or_query(self):
        with pytest.raises(UnsupportedPromptError) as raised:
            build_messages(
                [{"role": "user", "parts": [
                    {"type": "IMAGE_URL", "url": "https://x.example/secret.png?sig=shhh"},
                ]}],
                supports_images=True,
            )
        message = str(raised.value)
        assert "https://x.example/..." in message
        assert "shhh" not in message
        assert "secret.png" not in message

    @pytest.mark.parametrize("url", [
        "http://x.example/y.png",
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data/",
    ])
    def test_every_non_data_scheme_is_rejected(self, url):
        with pytest.raises(UnsupportedPromptError, match="does not accept remote image URLs"):
            build_messages(
                [{"role": "user", "parts": [{"type": "IMAGE_URI", "uri": url}]}],
                supports_images=True,
            )

    @pytest.mark.parametrize("inline_image,expected", [
        ("R0lGODlhAAA", "image/gif"),
        ("UklGRgAAAA", "image/webp"),
    ])
    def test_an_unsupported_inline_format_is_named_in_the_error(self, inline_image, expected):
        with pytest.raises(UnsupportedPromptError) as raised:
            build_messages(
                [{"role": "user", "parts": [
                    {"type": "IMAGE_INLINE", "inlineImage": inline_image},
                ]}],
                supports_images=True,
            )
        assert expected in str(raised.value)
        assert "PNG and JPEG" in str(raised.value)

    def test_an_unrecognised_inline_payload_is_rejected_not_relabelled_as_png(self):
        with pytest.raises(UnsupportedPromptError, match="unrecognised image format"):
            build_messages(
                [{"role": "user", "parts": [{"type": "IMAGE_INLINE", "inlineImage": "zzzz"}]}],
                supports_images=True,
            )

    @pytest.mark.parametrize("source", [
        "data:image/gif;base64,R0lGODlh",
        "data:;base64,iVBORw0KGgo",
        "data:text/html;base64,PHNjcmlwdD4=",
    ])
    def test_a_data_uri_declaring_an_unsupported_type_is_rejected(self, source):
        with pytest.raises(UnsupportedPromptError, match="PNG and JPEG"):
            build_messages(
                [{"role": "user", "parts": [{"type": "IMAGE_URL", "url": source}]}],
                supports_images=True,
            )

    def test_image_on_non_vision_model_raises_a_clear_error(self):
        with pytest.raises(UnsupportedPromptError, match="not configured for a vision-capable"):
            build_messages(
                [{"role": "user", "parts": [{"type": "IMAGE_INLINE", "inlineImage": "/9j/AAA"}]}],
                supports_images=False,
            )

    def test_image_on_non_user_role_raises(self):
        with pytest.raises(UnsupportedPromptError, match="only accepts image inputs on user"):
            build_messages(
                [{"role": "assistant", "parts": [{"type": "IMAGE_INLINE", "inlineImage": "/9j/A"}]}],
                supports_images=True,
            )

    def test_unknown_part_type_is_skipped(self):
        messages = build_messages([{"role": "user", "parts": [
            {"type": "TEXT", "text": "hi"},
            {"type": "AUDIO_INLINE", "inlineAudio": "..."},
        ]}])
        assert messages[0]["content"] == [{"type": "text", "text": "hi"}]


class TestBuildPayload:
    QUERY = {"messages": [{"role": "user", "content": "Hi"}]}

    def test_settings_are_renamed_to_cerebras_keys(self):
        payload = build_payload("gpt-oss-120b", self.QUERY, {
            "temperature": 0.2,
            "maxOutputTokens": 512,
            "topP": 0.9,
            "stopSequences": ["\n\n"],
        })
        assert payload["model"] == "gpt-oss-120b"
        assert payload["max_completion_tokens"] == 512
        assert payload["top_p"] == 0.9
        assert payload["stop"] == ["\n\n"]
        # CerebrasClient owns the stream flag so it cannot disagree with the read mode.
        assert "stream" not in payload

    def test_zero_temperature_is_not_dropped(self):
        payload = build_payload("gpt-oss-120b", self.QUERY, {"temperature": 0})
        assert payload["temperature"] == 0

    def test_tool_choice_variants(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        assert build_payload("m", self.QUERY, {"tools": tools, "toolChoice": {"type": "AUTO"}})[
            "tool_choice"] == "auto"
        assert build_payload("m", self.QUERY, {
            "tools": tools, "toolChoice": {"type": "FUNCTION", "name": "f"}
        })["tool_choice"] == {"type": "function", "function": {"name": "f"}}

    def test_json_schema_response_format(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}},
                  "additionalProperties": False}
        payload = build_payload("m", self.QUERY, {"responseFormat": {"type": "json", "schema": schema}})
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["schema"] == schema
        assert payload["response_format"]["json_schema"]["strict"] is True

    def _json_schema(self, schema):
        return build_payload(
            "m", self.QUERY, {"responseFormat": {"type": "json", "schema": schema}}
        )["response_format"]["json_schema"]

    def test_additional_properties_is_injected_where_dss_omits_it(self):
        """Strict decoding 400s without it: "'additionalProperties' is required
        to be supplied and set to false"."""
        built = self._json_schema({"type": "object", "properties": {"a": {"type": "string"}}})
        assert built["schema"]["additionalProperties"] is False
        assert built["strict"] is True

    def test_additional_properties_is_injected_at_every_depth(self):
        built = self._json_schema({
            "type": "object",
            "properties": {
                "nested": {"type": "object", "properties": {"b": {"type": "string"}}},
                "listed": {"type": "array", "items": {
                    "type": "object", "properties": {"c": {"type": "string"}}}},
                "either": {"anyOf": [{"type": "object", "properties": {"d": {"type": "string"}}}]},
            },
            "$defs": {"Ref": {"type": "object", "properties": {"e": {"type": "string"}}}},
        })
        schema = built["schema"]
        properties = schema["properties"]
        assert schema["additionalProperties"] is False
        assert properties["nested"]["additionalProperties"] is False
        assert properties["listed"]["items"]["additionalProperties"] is False
        assert properties["either"]["anyOf"][0]["additionalProperties"] is False
        assert schema["$defs"]["Ref"]["additionalProperties"] is False

    def test_an_explicit_false_is_left_alone(self):
        built = self._json_schema({"type": "object", "properties": {}, "additionalProperties": False})
        assert built["schema"]["additionalProperties"] is False
        assert built["strict"] is True

    def test_a_schema_wanting_extra_properties_is_sent_unstrict(self):
        """Overriding the author to force strict would silently change the contract."""
        schema = {"type": "object", "properties": {"a": {"type": "string"}},
                  "additionalProperties": True}
        built = self._json_schema(schema)
        assert "strict" not in built
        assert built["schema"] == schema

    def test_extra_properties_nested_deep_also_drops_strict(self):
        built = self._json_schema({
            "type": "object",
            "properties": {"n": {"type": "object", "properties": {}, "additionalProperties": True}},
        })
        assert "strict" not in built

    def test_a_property_named_properties_is_not_mistaken_for_a_schema(self):
        built = self._json_schema({
            "type": "object",
            "properties": {"properties": {"type": "string"}},
        })
        # The string property must stay a string, not gain additionalProperties.
        assert built["schema"]["properties"]["properties"] == {"type": "string"}

    def test_the_callers_schema_is_not_mutated(self):
        """DSS may reuse the settings dict across a retry or a second call."""
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        self._json_schema(schema)
        assert schema == {"type": "object", "properties": {"a": {"type": "string"}}}

    def test_plain_json_response_format(self):
        payload = build_payload("m", self.QUERY, {"responseFormat": {"type": "json"}})
        assert payload["response_format"] == {"type": "json_object"}

    def test_response_format_is_dropped_when_tools_are_present(self):
        # gpt-oss-120b rejects requests carrying both fields.
        payload = build_payload("gpt-oss-120b", self.QUERY, {
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "responseFormat": {"type": "json"},
        })
        assert "response_format" not in payload
        assert "tools" in payload

    def test_reasoning_effort(self):
        payload = build_payload("m", self.QUERY, {}, reasoning_effort="high")
        assert payload["reasoning_effort"] == "high"

    def test_extra_body_is_merged(self):
        payload = build_payload("m", self.QUERY, {}, extra_body={"prompt_cache_key": "abc"})
        assert payload["prompt_cache_key"] == "abc"


class TestParseCompletion:
    def test_text_and_tokens(self):
        result = parse_completion({
            "choices": [{"message": {"role": "assistant", "content": "Hello!"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        assert result["text"] == "Hello!"
        assert result["promptTokens"] == 10
        assert result["completionTokens"] == 5
        assert result["finishReason"] == "stop"

    def test_completion_tokens_derived_from_total(self):
        result = parse_completion({
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 8, "total_tokens": 20},
        })
        assert result["completionTokens"] == 12

    def test_tool_calls_force_the_finish_reason(self):
        result = parse_completion({"choices": [{
            "message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "f", "arguments": '{"a":1}'}}]},
            "finish_reason": "stop",
        }]})
        assert result["finishReason"] == "tool_calls"
        assert result["toolCalls"][0]["id"] == "c1"
        assert result["text"] == ""

    def test_reasoning_is_hidden_by_default(self):
        response = {"choices": [{"message": {"content": "42", "reasoning": "thinking hard"}}]}
        assert parse_completion(response)["text"] == "42"
        assert parse_completion(response)["_reasoning"] == "thinking hard"

    def test_reasoning_can_be_prefixed_to_the_text(self):
        result = parse_completion(
            {"choices": [{"message": {"content": "42", "reasoning": "thinking hard"}}]},
            include_reasoning_in_text=True,
        )
        assert result["text"] == "<thinking>\nthinking hard\n</thinking>\n\n42"

    def test_empty_response_does_not_crash(self):
        result = parse_completion({})
        assert result["text"] == ""
        assert result["toolCalls"] == []


class TestStreaming:
    def test_text_chunks_then_footer(self):
        lines = [
            _sse({"choices": [{"delta": {"role": "assistant"}}]}),
            _sse({"choices": [{"delta": {"content": "Hel"}}]}),
            _sse({"choices": [{"delta": {"content": "lo"}}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 3, "completion_tokens": 2}}),
            "data: [DONE]",
        ]
        chunks = list(stream_to_dataiku_chunks(lines))
        assert chunks[0] == {"text": "Hel"}
        assert chunks[1] == {"text": "lo"}
        footer = chunks[-1]["footer"]
        assert footer["promptTokens"] == 3
        assert footer["completionTokens"] == 2
        assert footer["finishReason"] == "stop"

    def test_tool_call_arguments_are_reassembled_across_chunks(self):
        lines = [
            _sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": ""}}]}}]}),
            _sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"city":'}}]}}]}),
            _sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ' "Sunnyvale"}'}}]}}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            "data: [DONE]",
        ]
        chunks = list(stream_to_dataiku_chunks(lines))
        tool_chunk = next(c for c in chunks if "toolCalls" in c)
        call = tool_chunk["toolCalls"][0]
        assert call["id"] == "call_1"
        assert call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"]) == {"city": "Sunnyvale"}
        assert chunks[-1]["footer"]["finishReason"] == "tool_calls"

    def test_parallel_tool_calls_keep_their_order(self):
        lines = [
            _sse({"choices": [{"delta": {"tool_calls": [
                {"index": 1, "id": "b", "function": {"name": "second", "arguments": "{}"}},
                {"index": 0, "id": "a", "function": {"name": "first", "arguments": "{}"}}]}}]}),
            "data: [DONE]",
        ]
        chunks = list(stream_to_dataiku_chunks(lines))
        names = [c["function"]["name"] for c in
                 next(c for c in chunks if "toolCalls" in c)["toolCalls"]]
        assert names == ["first", "second"]

    def test_reasoning_deltas_are_wrapped_when_requested(self):
        lines = [
            _sse({"choices": [{"delta": {"reasoning": "step 1"}}]}),
            _sse({"choices": [{"delta": {"content": "answer"}}]}),
            "data: [DONE]",
        ]
        text = "".join(c["text"] for c in stream_to_dataiku_chunks(
            lines, include_reasoning_in_text=True) if "text" in c)
        assert text == "<thinking>\nstep 1\n</thinking>\n\nanswer"

    def test_reasoning_deltas_are_dropped_by_default(self):
        lines = [
            _sse({"choices": [{"delta": {"reasoning": "step 1"}}]}),
            _sse({"choices": [{"delta": {"content": "answer"}}]}),
            "data: [DONE]",
        ]
        text = "".join(c["text"] for c in stream_to_dataiku_chunks(lines) if "text" in c)
        assert text == "answer"

    def test_keepalives_and_garbage_are_ignored(self):
        lines = ["", ": keep-alive", "data: ", "data: not-json",
                 _sse({"choices": [{"delta": {"content": "ok"}}]}), "data: [DONE]"]
        chunks = list(stream_to_dataiku_chunks(lines))
        assert chunks[0] == {"text": "ok"}

    def test_byte_lines_are_decoded(self):
        # requests.iter_lines() yields bytes.
        lines = [_sse({"choices": [{"delta": {"content": "ok"}}]}).encode("utf-8")]
        assert list(stream_to_dataiku_chunks(lines))[0] == {"text": "ok"}

    def test_footer_is_always_emitted(self):
        assert list(stream_to_dataiku_chunks([]))[-1]["footer"]["finishReason"] == "stop"


class TestHelpers:
    def test_parse_stream_line(self):
        assert parse_stream_line("event: ping") is None
        assert parse_stream_line("data: [DONE]") == {"_done": True}
        assert parse_stream_line('data: {"a": 1}') == {"a": 1}

    def test_detect_mime_type(self):
        assert detect_mime_type("/9j/4AAQ") == "image/jpeg"
        assert detect_mime_type("iVBORw0KGgo") == "image/png"
        assert detect_mime_type("R0lGODlh") == "image/gif"
        # Unrecognised is None, not a PNG guess: the caller rejects rather than relabels.
        assert detect_mime_type("zzz") is None

    def test_data_uri_mime_type(self):
        assert data_uri_mime_type("data:image/png;base64,AAAA") == "image/png"
        assert data_uri_mime_type("data:IMAGE/JPEG;base64,AAAA") == "image/jpeg"
        assert data_uri_mime_type("data:;base64,AAAA") == ""

    def test_coerce_extra_body_parses_json_strings(self):
        coerced = coerce_extra_body({"parallel_tool_calls": "true", "n": "2", "key": "abc"})
        assert coerced == {"parallel_tool_calls": True, "n": 2, "key": "abc"}

    def test_normalize_tool_calls_generates_missing_ids(self):
        calls = normalize_tool_calls([{"function": {"name": "f"}}])
        assert calls[0]["id"] == "call_0"
        assert calls[0]["function"]["arguments"] == "{}"
