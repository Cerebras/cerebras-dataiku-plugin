"""Tests for the HTTP layer: retry policy, error mapping, header construction."""

import json
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-lib"))

from dku_cerebras.client import CerebrasAPIError, CerebrasClient  # noqa: E402


class FakeResponse(object):
    def __init__(self, status_code=200, body=None, headers=None, text=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(body or {})
        self.closed = False

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Keep the backoff logic under test but drop the wall-clock cost."""
    monkeypatch.setattr("dku_cerebras.client.time.sleep", lambda seconds: None)


def _client(**kwargs):
    kwargs.setdefault("api_key", "csk-test")
    return CerebrasClient(**kwargs)


def _record_posts(monkeypatch, client, responses):
    """Make session.post return the queued responses in order, recording its kwargs."""
    calls = []
    queue = list(responses)

    def fake_post(url, **kwargs):
        calls.append(dict(url=url, **kwargs))
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(client.session, "post", fake_post)
    return calls


class TestConstruction:
    def test_api_key_is_required(self):
        with pytest.raises(ValueError, match="No Cerebras API key configured"):
            CerebrasClient(api_key="")

    def test_headers(self):
        assert _client()._headers()["Authorization"] == "Bearer csk-test"

    def test_negative_retries_are_clamped(self):
        assert _client(max_retries=-5).max_retries == 0


class TestSuccess:
    def test_returns_parsed_json_and_hits_the_right_url(self, monkeypatch):
        client = _client()
        calls = _record_posts(monkeypatch, client, [FakeResponse(200, {"choices": []})])
        assert client.chat_completion({"model": "m"}) == {"choices": []}
        assert calls[0]["url"] == "https://api.cerebras.ai/v1/chat/completions"
        assert calls[0]["stream"] is False
        assert json.loads(calls[0]["data"]) == {"model": "m", "stream": False}

    def test_streaming_returns_the_live_response_and_asks_for_sse(self, monkeypatch):
        client = _client()
        response = FakeResponse(200)
        calls = _record_posts(monkeypatch, client, [response])
        assert client.stream_chat_completion({"model": "m"}) is response
        assert calls[0]["headers"]["Accept"] == "text/event-stream"
        assert calls[0]["stream"] is True
        assert json.loads(calls[0]["data"])["stream"] is True

    def test_the_client_overrides_a_conflicting_stream_flag(self, monkeypatch):
        """The payload field and the HTTP read mode can never disagree."""
        client = _client()
        calls = _record_posts(monkeypatch, client, [FakeResponse(200, {}), FakeResponse(200)])

        client.chat_completion({"model": "m", "stream": True})
        assert json.loads(calls[0]["data"])["stream"] is False
        assert calls[0]["stream"] is False

        client.stream_chat_completion({"model": "m", "stream": False})
        assert json.loads(calls[1]["data"])["stream"] is True
        assert calls[1]["stream"] is True

    def test_the_callers_payload_is_not_mutated(self, monkeypatch):
        client = _client()
        _record_posts(monkeypatch, client, [FakeResponse(200, {})])
        payload = {"model": "m"}
        client.chat_completion(payload)
        assert payload == {"model": "m"}

    def test_dedicated_endpoint_base_url(self, monkeypatch):
        client = _client(base_url="https://my-endpoint.cerebras.ai/v1")
        calls = _record_posts(monkeypatch, client, [FakeResponse(200, {})])
        client.chat_completion({"model": "m"})
        assert calls[0]["url"] == "https://my-endpoint.cerebras.ai/v1/chat/completions"


class TestErrors:
    def test_client_errors_are_not_retried(self, monkeypatch):
        client = _client(max_retries=3)
        calls = _record_posts(monkeypatch, client, [
            FakeResponse(401, {"error": {"message": "Wrong API Key"}})
        ])
        with pytest.raises(CerebrasAPIError, match="401: Wrong API Key") as excinfo:
            client.chat_completion({"model": "m"})
        assert excinfo.value.status_code == 401
        assert len(calls) == 1

    def test_rate_limits_are_retried_then_surfaced(self, monkeypatch):
        client = _client(max_retries=2)
        calls = _record_posts(monkeypatch, client, [
            FakeResponse(429, {"error": {"message": "slow down"}}, {"Retry-After": "0.01"}),
            FakeResponse(429, {"error": {"message": "slow down"}}, {"Retry-After": "0.01"}),
            FakeResponse(429, {"error": {"message": "slow down"}}, {"Retry-After": "0.01"}),
        ])
        with pytest.raises(CerebrasAPIError, match="429"):
            client.chat_completion({"model": "m"})
        assert len(calls) == 3  # the initial attempt plus two retries

    def test_a_transient_500_then_success(self, monkeypatch):
        client = _client(max_retries=2)
        calls = _record_posts(monkeypatch, client, [
            FakeResponse(503, {"error": {"message": "unavailable"}}),
            FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]}),
        ])
        result = client.chat_completion({"model": "m"})
        assert result["choices"][0]["message"]["content"] == "ok"
        assert len(calls) == 2

    def test_connection_errors_are_retried(self, monkeypatch):
        client = _client(max_retries=1)
        calls = _record_posts(monkeypatch, client, [
            requests.exceptions.ConnectionError("reset by peer"),
            FakeResponse(200, {"choices": []}),
        ])
        assert client.chat_completion({"model": "m"}) == {"choices": []}
        assert len(calls) == 2

    def test_connection_errors_surface_with_a_readable_message(self, monkeypatch):
        client = _client(max_retries=0)
        _record_posts(monkeypatch, client, [requests.exceptions.Timeout("timed out")])
        with pytest.raises(CerebrasAPIError, match="Could not reach the Cerebras API"):
            client.chat_completion({"model": "m"})

    def test_non_json_error_bodies_still_produce_a_message(self, monkeypatch):
        client = _client(max_retries=0)
        _record_posts(monkeypatch, client, [
            FakeResponse(502, body=None, text="<html>Bad Gateway</html>")
        ])
        with pytest.raises(CerebrasAPIError, match="Bad Gateway"):
            client.chat_completion({"model": "m"})

    def test_retried_responses_are_closed(self, monkeypatch):
        client = _client(max_retries=1)
        failing = FakeResponse(500, {"error": {"message": "boom"}})
        _record_posts(monkeypatch, client, [failing, FakeResponse(200, {})])
        client.chat_completion({"model": "m"})
        assert failing.closed is True


class TestNoOutboundImageFetching:
    def test_the_client_exposes_no_image_downloader(self):
        # Images must arrive inlined in the prompt. The client deliberately has no
        # way to fetch a prompt-supplied URL -- see mapping._build_content_parts.
        assert not hasattr(_client(), "fetch_image_as_data_uri")
