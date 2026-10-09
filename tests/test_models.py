"""Tests for GET /v1/models and the local model-capability knowledge.

`list_models()` no longer backs a dropdown -- DSS does not run choices callbacks for
python-llms components -- but it is still how `tests/live_check.py --only models` shows
you what a key can reach, so it stays covered.
"""

import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "python-lib"))

from dku_cerebras.client import CerebrasAPIError, CerebrasClient  # noqa: E402
from dku_cerebras.models import VISION_MODELS, model_label  # noqa: E402


def _client_returning(monkeypatch, body, status_code=200, base_url=None):
    """A client whose GET returns `body`, recording the URL it was given."""
    client = CerebrasClient(api_key="csk-test", base_url=base_url)
    captured = {}

    class FakeResponse(object):
        def __init__(self):
            self.status_code = status_code
            self.headers = {}
            self.text = str(body)

        def json(self):
            return body

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        return FakeResponse()

    monkeypatch.setattr(client.session, "get", fake_get)
    return client, captured


class TestModelLabels:
    def test_vision_models_are_flagged(self):
        assert model_label("qwen-3.8-27b") == "qwen-3.8-27b (vision)"
        assert model_label("gemma-4-31b") == "gemma-4-31b (vision)"
        assert model_label("gpt-oss-120b") == "gpt-oss-120b"

    def test_an_unknown_model_is_not_assumed_to_do_vision(self):
        assert "some-future-model" not in VISION_MODELS
        assert model_label("some-future-model") == "some-future-model"


class TestListModels:
    def test_parses_the_models_response(self, monkeypatch):
        # Placeholder ids: this covers parsing and ordering, not the real catalog,
        # which would only go stale as models come and go.
        client, captured = _client_returning(monkeypatch, {"object": "list", "data": [
            {"id": "model-two", "object": "model", "owned_by": "Cerebras"},
            {"id": "model-one", "object": "model", "owned_by": "Cerebras"},
        ]})
        assert client.list_models() == ["model-two", "model-one"]
        assert captured["url"] == "https://api.cerebras.ai/v1/models"
        assert captured["headers"]["Authorization"] == "Bearer csk-test"

    def test_entries_without_an_id_are_skipped(self, monkeypatch):
        client, _ = _client_returning(
            monkeypatch, {"data": [{"id": "a"}, {"object": "model"}]}
        )
        assert client.list_models() == ["a"]

    def test_an_empty_catalog_returns_an_empty_list(self, monkeypatch):
        client, _ = _client_returning(monkeypatch, {"data": []})
        assert client.list_models() == []

    def test_a_missing_data_key_returns_an_empty_list(self, monkeypatch):
        client, _ = _client_returning(monkeypatch, {"object": "list"})
        assert client.list_models() == []

    def test_http_errors_raise(self, monkeypatch):
        client, _ = _client_returning(
            monkeypatch, {"error": {"message": "Wrong API Key"}}, status_code=401
        )
        with pytest.raises(CerebrasAPIError, match="401: Wrong API Key"):
            client.list_models()

    def test_a_dedicated_endpoint_url_is_respected(self, monkeypatch):
        client, captured = _client_returning(
            monkeypatch, {"data": [{"id": "my-tuned-model"}]},
            base_url="https://my-endpoint/v1",
        )
        assert client.list_models() == ["my-tuned-model"]
        assert captured["url"] == "https://my-endpoint/v1/models"
