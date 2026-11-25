"""Tests for the DSS-facing BaseLLM implementation.

`dataiku` is not importable outside DSS and there is no live API here, so both are
stubbed. What this actually verifies is the wiring: that the param names in llm.json
are the ones llm.py reads, and that process()/process_stream() honour them.
"""

import json
import os
import sys
import time
import types

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "python-lib"))
sys.path.insert(0, os.path.join(HERE, "..", "python-llms", "cerebras-chat"))


def _install_dataiku_stub():
    """Minimal stand-in for the dataiku.llm.python module DSS provides."""
    if "dataiku.llm.python" in sys.modules:
        return
    dataiku = types.ModuleType("dataiku")
    llm = types.ModuleType("dataiku.llm")
    python = types.ModuleType("dataiku.llm.python")

    class BaseLLM(object):
        pass

    python.BaseLLM = BaseLLM
    python.BaseEmbeddingModel = type("BaseEmbeddingModel", (object,), {})
    dataiku.llm = llm
    llm.python = python
    sys.modules.update({"dataiku": dataiku, "dataiku.llm": llm, "dataiku.llm.python": python})


_install_dataiku_stub()

from llm import CerebrasLLM  # noqa: E402


class FakeSpan(object):
    def __init__(self):
        self.attributes = {}
        self.start = self.finish = None

    def __enter__(self):
        self.begin(int(time.time() * 1000))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end(int(time.time() * 1000))

    def begin(self, timestamp):
        self.start = timestamp

    def end(self, timestamp):
        assert self.start is not None and self.finish is None
        self.finish = timestamp


class FakeTrace(object):
    def __init__(self):
        self.spans = []

    def subspan(self, name):
        span = FakeSpan()
        self.spans.append((name, span))
        return span


class FakeStreamResponse(object):
    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def iter_lines(self):
        for line in self._lines:
            yield line.encode("utf-8")

    def close(self):
        self.closed = True


def _config(**overrides):
    """A config dict shaped exactly like what llm.json produces."""
    config = {
        "credential_mode": "SHARED",
        "shared_key": {"api_key": "csk-test"},
        "base_url": "https://api.cerebras.ai/v1",
        "model": "gpt-oss-120b",
        "reasoning_effort": "",
        "include_reasoning_in_text": False,
        "request_timeout": 300,
        "max_retries": 3,
        "extra_body": {},
    }
    config.update(overrides)
    return config


def _llm(monkeypatch, response=None, stream_lines=None, **config_overrides):
    """Build a configured CerebrasLLM whose HTTP call is replaced by a recorder."""
    llm = CerebrasLLM()
    llm.set_config(_config(**config_overrides), {})

    sent = {}

    def fake_chat_completion(payload):
        sent["payload"] = payload
        sent["method"] = "chat_completion"
        return response or {}

    def fake_stream_chat_completion(payload):
        sent["payload"] = payload
        sent["method"] = "stream_chat_completion"
        return FakeStreamResponse(stream_lines or [])

    monkeypatch.setattr(llm.client, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(llm.client, "stream_chat_completion", fake_stream_chat_completion)
    return llm, sent


def _expected_reasoning(text):
    return {
        "id": "chatcmpl-1",
        "type": "REASONING",
        "parts": [{"type": "TEXT", "index": 0, "text": text}],
    }


class TestSetConfig:
    def test_reads_the_credential_from_the_preset_and_the_url_from_the_connection(self):
        llm = CerebrasLLM()
        llm.set_config(_config(), {})
        assert llm.model == "gpt-oss-120b"
        assert llm.client.api_key == "csk-test"
        assert llm.client.base_url == "https://api.cerebras.ai/v1"
        assert llm.supports_images is False

    def test_vision_model_enables_images_without_the_checkbox(self):
        llm = CerebrasLLM()
        llm.set_config(_config(model="qwen-3.8-27b"), {})
        assert llm.supports_images is True

    def test_a_free_text_model_id_is_trimmed(self):
        llm = CerebrasLLM()
        llm.set_config(_config(model="  my-tuned-model  "), {})
        assert llm.model == "my-tuned-model"
        assert llm.supports_images is False

    def test_the_checkbox_forces_images_on_for_an_unrecognised_model(self):
        # Any model the plugin has no vision information about.
        llm = CerebrasLLM()
        llm.set_config(_config(model="some-future-vision-model", supports_images=True), {})
        assert llm.supports_images is True

    def test_a_blank_model_fails_fast(self):
        llm = CerebrasLLM()
        with pytest.raises(ValueError, match="No model configured"):
            llm.set_config(_config(model="   "), {})

    def test_an_unconfigured_shared_preset_fails_fast(self):
        llm = CerebrasLLM()
        with pytest.raises(ValueError, match="No shared Cerebras API key"):
            llm.set_config(_config(shared_key={}), {})

    def test_an_individual_key_is_used_when_that_mode_is_selected(self):
        llm = CerebrasLLM()
        llm.set_config(_config(credential_mode="PER_USER",
                               user_key={"api_key": "csk-mine"}), {})
        assert llm.client.api_key == "csk-mine"

    def test_an_unfilled_individual_credential_fails_fast(self):
        llm = CerebrasLLM()
        with pytest.raises(ValueError, match="Profile and settings > Credentials"):
            llm.set_config(_config(credential_mode="PER_USER", user_key={}), {})

    def test_dedicated_endpoint_url_is_used_and_normalised(self):
        llm = CerebrasLLM()
        llm.set_config(_config(base_url="https://my-endpoint.cerebras.ai/v1/"), {})
        assert llm.client.base_url == "https://my-endpoint.cerebras.ai/v1"


class TestProcess:
    RESPONSE = {
        "choices": [{"message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        "time_info": {"total_time": 0.04},
    }

    def test_returns_the_dss_response_contract(self, monkeypatch):
        llm, sent = _llm(monkeypatch, response=self.RESPONSE)
        result = llm.process({"messages": [{"role": "user", "content": "ping"}]},
                             {"temperature": 0.1}, FakeTrace())

        assert result["text"] == "pong"
        assert result["promptTokens"] == 10
        assert result["completionTokens"] == 2
        assert result["finishReason"] == "stop"
        assert "artifacts" not in result
        assert "logProbs" not in result
        # `_reasoning` is internal and must not leak into the DSS response.
        assert "_reasoning" not in result

        # The non-streaming path must use the non-streaming client method; the client
        # owns the payload's stream flag, so it is not set here.
        assert sent["method"] == "chat_completion"
        assert sent["payload"]["model"] == "gpt-oss-120b"
        assert "stream" not in sent["payload"]
        assert sent["payload"]["temperature"] == 0.1

    def test_returns_token_log_probs_and_alternatives(self, monkeypatch):
        choice = {
            "message": {"content": "Yes!"},
            "logprobs": {"content": [
                {"token": "Yes", "logprob": -0.2, "bytes": [89, 101, 115], "top_logprobs": [
                    {"token": "Yes", "logprob": -0.2}, {"token": "No", "logprob": -1.8},
                ]},
                {"token": "!", "logprob": 0.0, "top_logprobs": []},
            ]},
        }
        llm, sent = _llm(monkeypatch, response={"choices": [choice]})
        result = llm.process({"messages": []}, {"logProbs": True, "topLogProbs": 2}, FakeTrace())

        assert sent["payload"]["logprobs"] is True
        assert sent["payload"]["top_logprobs"] == 2
        assert result["text"] == "Yes!"
        assert result["logProbs"] == [
            {"token": "Yes", "logProb": -0.2, "topLogProbs": [
                {"token": "Yes", "logProb": -0.2}, {"token": "No", "logProb": -1.8},
            ]},
            {"token": "!", "logProb": 0.0, "topLogProbs": []},
        ]

    def test_connection_settings_reach_the_payload(self, monkeypatch):
        llm, sent = _llm(monkeypatch, response=self.RESPONSE, reasoning_effort="high",
                         extra_body={"prompt_cache_key": "team-a"})
        llm.process({"messages": [{"role": "user", "content": "ping"}]}, {}, FakeTrace())
        assert sent["payload"]["reasoning_effort"] == "high"
        # Anything without a dedicated field still reaches the API via extra payload fields.
        assert sent["payload"]["prompt_cache_key"] == "team-a"

    @pytest.mark.parametrize("inline, expected_text", [
        (False, "42"),
        (True, "<thinking>\ndeep thought\n</thinking>\n\n42"),
    ])
    def test_returns_reasoning_artifact_and_records_trace(self, monkeypatch, inline, expected_text):
        response = {"id": "chatcmpl-1",
                    "choices": [{"message": {"content": "42", "reasoning": "deep thought"}}]}
        llm, _ = _llm(monkeypatch, response=response, include_reasoning_in_text=inline)
        trace = FakeTrace()
        result = llm.process({"messages": [{"role": "user", "content": "?"}]}, {}, trace)

        assert result["text"] == expected_text
        assert result["artifacts"] == [_expected_reasoning("deep thought")]
        assert "_reasoning" not in result
        assert trace.spans[0][1].finish is not None
        assert trace.spans[0][0] == "Cerebras chat completion"
        assert trace.spans[0][1].attributes["reasoning"] == "deep thought"


class TestProcessStream:
    LINES = [
        'data: {"choices": [{"delta": {"content": "po"}}]}',
        'data: {"choices": [{"delta": {"content": "ng"}}]}',
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}],'
        ' "usage": {"prompt_tokens": 4, "completion_tokens": 2}}',
        "data: [DONE]",
    ]

    def test_yields_text_then_a_footer(self, monkeypatch):
        llm, sent = _llm(monkeypatch, stream_lines=self.LINES)
        chunks = list(llm.process_stream({"messages": [{"role": "user", "content": "ping"}]},
                                         {}, FakeTrace()))

        assert "".join(c["text"] for c in chunks if "text" in c) == "pong"
        assert all("artifacts" not in chunk for chunk in chunks)
        footer = chunks[-1]["footer"]
        assert footer["promptTokens"] == 4
        assert footer["finishReason"] == "stop"
        assert sent["method"] == "stream_chat_completion"

    @pytest.mark.parametrize("inline, expected_text", [
        (False, "42"),
        (True, "<thinking>\ndeep thought\n</thinking>\n\n42"),
    ])
    def test_streams_reasoning_artifact_deltas(self, monkeypatch, inline, expected_text):
        lines = [
            'data: {"id":"chatcmpl-1","choices":[{"delta":{"reasoning":"deep"}}]}',
            'data: {"id":"chatcmpl-1","choices":[{"delta":{"reasoning":" thought","content":"42"}}]}',
            "data: [DONE]",
        ]
        llm, _ = _llm(monkeypatch, stream_lines=lines, include_reasoning_in_text=inline)
        chunks = list(llm.process_stream({"messages": []}, {}, FakeTrace()))

        assert [chunk for chunk in chunks if "artifacts" in chunk] == [
            {"type": "content", "artifacts": [_expected_reasoning("deep")]},
            {"type": "content", "artifacts": [_expected_reasoning(" thought")]},
        ]
        assert "".join(c["text"] for c in chunks if "text" in c) == expected_text
        assert chunks[-1]["footer"]["finishReason"] == "stop"

    def test_streams_log_probs_on_each_content_chunk(self, monkeypatch):
        choices = [
            {"delta": {"content": "Yes"}, "logprobs": {"content": [
                {"token": "Yes", "logprob": -0.2, "top_logprobs": [{"token": "Yes", "logprob": -0.2}]}
            ]}},
            {"delta": {"content": "!"}, "logprobs": None},
            {"delta": {}, "logprobs": {"content": [{"token": "!", "logprob": 0.0}]}},
            {"delta": {}, "logprobs": {"content": None}, "finish_reason": "stop"},
        ]
        lines = ["data: " + json.dumps({"choices": [choice]}) for choice in choices]
        llm, _ = _llm(monkeypatch, stream_lines=lines + ["data: [DONE]"])
        stream = llm.process_stream({"messages": []}, {"logProbs": True}, FakeTrace())

        assert list(stream) == [
            {"text": "Yes", "logProbs": [
                {"token": "Yes", "logProb": -0.2, "topLogProbs": [{"token": "Yes", "logProb": -0.2}]}
            ]},
            {"text": "!"},
            {"text": "", "logProbs": [{"token": "!", "logProb": 0.0, "topLogProbs": []}]},
            {"footer": {"promptTokens": 0, "completionTokens": 0, "finishReason": "stop"}},
        ]

    def test_footer_is_recorded_on_the_trace(self, monkeypatch):
        llm, _ = _llm(monkeypatch, stream_lines=self.LINES)
        trace = FakeTrace()
        stream = llm.process_stream({"messages": []}, {}, trace)
        assert next(stream) == {"text": "po"}
        name, span = trace.spans[0]
        assert name == "Cerebras chat completion (streaming)"
        assert span.start is not None and span.finish is None
        assert next(stream) == {"text": "ng"}
        assert "footer" in next(stream)
        assert span.finish is not None
        assert trace.spans[0][1].attributes["completionTokens"] == 2

    def test_the_response_is_closed_even_if_the_consumer_stops_early(self, monkeypatch):
        responses = []

        llm = CerebrasLLM()
        llm.set_config(_config(), {})

        def fake_stream_chat_completion(payload):
            response = FakeStreamResponse(self.LINES)
            responses.append(response)
            return response

        monkeypatch.setattr(llm.client, "stream_chat_completion", fake_stream_chat_completion)

        trace = FakeTrace()
        generator = llm.process_stream({"messages": [{"role": "user", "content": "ping"}]},
                                       {}, trace)
        next(generator)
        generator.close()
        assert responses[0].closed is True
        assert trace.spans[0][1].finish is not None


@pytest.mark.parametrize("streaming", [False, True])
def test_provider_failure_closes_span(monkeypatch, streaming):
    llm = CerebrasLLM()
    llm.set_config(_config(), {})
    trace = FakeTrace()

    def fail(payload):
        raise RuntimeError("provider failed")

    method = "stream_chat_completion" if streaming else "chat_completion"
    monkeypatch.setattr(llm.client, method, fail)
    with pytest.raises(RuntimeError, match="provider failed"):
        if streaming:
            next(llm.process_stream({"messages": []}, {}, trace))
        else:
            llm.process({"messages": []}, {}, trace)
    assert trace.spans[0][1].finish is not None


class TestToolRoundTrip:
    def test_dss_tool_history_is_translated_for_the_api(self, monkeypatch):
        """The shape DSS sends back after running a tool must survive translation."""
        llm, sent = _llm(monkeypatch, response={"choices": [{"message": {"content": "21C"}}]})
        query = {"messages": [
            {"role": "user", "content": "weather in Sunnyvale?"},
            {"role": "assistant", "toolCalls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Sunnyvale"}}}]},
            {"role": "tool", "toolOutputs": [{"callId": "call_1", "output": "21C and sunny"}]},
        ]}
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        llm.process(query, {"tools": tools, "toolChoice": {"type": "AUTO"}}, FakeTrace())

        messages = sent["payload"]["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Sunnyvale"}'
        assert messages[2]["tool_call_id"] == "call_1"
        assert sent["payload"]["tool_choice"] == "auto"
        assert json.dumps(sent["payload"])  # the whole payload must be JSON-serialisable
