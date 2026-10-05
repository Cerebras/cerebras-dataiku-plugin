"""Thin HTTP client for the Cerebras Inference API.

Deliberately built on `requests` only: it is already in the DSS builtin environment,
so this plugin needs no code environment at all and works unchanged against
api.cerebras.ai, a dedicated endpoint, or a gateway in front of either.
"""

import base64
import json
import logging
import random
import time

import requests

from . import DEFAULT_BASE_URL

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"

# 429 is a rate limit, 5xx are transient on the server side. Everything else is a
# client error that retrying cannot fix.
RETRYABLE_STATUS_CODES = frozenset([408, 409, 429, 500, 502, 503, 504])

_MAX_BACKOFF_SECONDS = 30.0
_IMAGE_DOWNLOAD_TIMEOUT = 30


class CerebrasAPIError(Exception):
    """A non-retryable or retry-exhausted error from the Cerebras API."""

    def __init__(self, message, status_code=None, request_id=None):
        super(CerebrasAPIError, self).__init__(message)
        self.status_code = status_code
        self.request_id = request_id


def _extract_error_message(response):
    """Pull the human-readable message out of a Cerebras error response."""
    try:
        body = response.json()
    except ValueError:
        return (response.text or "").strip()[:500]

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return error.get("message") or json.dumps(error)[:500]
        if isinstance(error, str):
            return error
        if body.get("message"):
            return body["message"]
    return json.dumps(body)[:500]


def _retry_delay(response, attempt):
    """Backoff in seconds, honouring Retry-After when the server sets it."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), _MAX_BACKOFF_SECONDS)
            except ValueError:
                pass
    # Exponential backoff with jitter, so parallel DSS activities do not sync up.
    return min(2.0 ** attempt, _MAX_BACKOFF_SECONDS) * (0.5 + random.random() / 2.0)


class CerebrasClient(object):
    def __init__(self, api_key, base_url=None, timeout=300, max_retries=3):
        if not api_key:
            raise ValueError(
                "No Cerebras API key configured. Set your key on the 'Cerebras API key' "
                "preset used by this connection."
            )
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout or 300
        self.max_retries = max(0, max_retries or 0)
        self.session = requests.Session()

    def _headers(self):
        return {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "dataiku-cerebras-plugin/1.0.0",
        }

    def chat_completion(self, payload):
        """POST /v1/chat/completions and return the parsed response body."""
        return self._post_with_retries(payload, stream=False).json()

    def stream_chat_completion(self, payload):
        """POST /v1/chat/completions as SSE, returning the live response.

        The body is unread: the caller iterates it and owns closing it.
        """
        return self._post_with_retries(payload, stream=True)

    def _post_with_retries(self, payload, stream):
        """Send the request, retrying transient failures, and return the Response.

        `stream` sets both the payload field and the HTTP read mode, so the two can
        never disagree.
        """
        payload = dict(payload, stream=stream)
        url = self.base_url + CHAT_COMPLETIONS_PATH
        headers = self._headers()
        if stream:
            headers["Accept"] = "text/event-stream"

        body = json.dumps(payload)
        last_error = None

        for attempt in range(self.max_retries + 1):
            response = None
            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    data=body,
                    stream=stream,
                    timeout=self.timeout,
                )
                if response.status_code < 400:
                    logger.info(
                        "Cerebras %s returned %s (request id %s)",
                        payload.get("model"), response.status_code,
                        response.headers.get("x-request-id"),
                    )
                    return response

                message = _extract_error_message(response)
                request_id = response.headers.get("x-request-id")
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise CerebrasAPIError(
                        "Cerebras API returned {}: {}".format(response.status_code, message),
                        status_code=response.status_code,
                        request_id=request_id,
                    )
                last_error = CerebrasAPIError(
                    "Cerebras API returned {}: {}".format(response.status_code, message),
                    status_code=response.status_code,
                    request_id=request_id,
                )

            except requests.exceptions.RequestException as error:
                # Connection resets, DNS failures and timeouts are all worth retrying.
                last_error = CerebrasAPIError("Could not reach the Cerebras API: {}".format(error))

            if attempt >= self.max_retries:
                break
            delay = _retry_delay(response, attempt)
            logger.warning(
                "Cerebras request failed (%s), retrying in %.1fs (attempt %d/%d)",
                last_error, delay, attempt + 1, self.max_retries,
            )
            if response is not None:
                response.close()
            time.sleep(delay)

        # last_error is always set by this point; the fallback only makes that provable.
        raise last_error or CerebrasAPIError("Cerebras request failed without an error")

    def list_models(self):
        """GET /v1/models, returning the model IDs available to this key.

        Used by tests/live_check.py; nothing in the plugin calls this at runtime, since
        the Model field is free text rather than a dropdown. Single-attempt by design:
        for an interactive caller a fast failure beats hanging through a retry cycle.
        """
        response = self.session.get(
            self.base_url + MODELS_PATH, headers=self._headers(), timeout=self.timeout
        )
        if response.status_code >= 400:
            raise CerebrasAPIError(
                "Cerebras API returned {}: {}".format(
                    response.status_code, _extract_error_message(response)
                ),
                status_code=response.status_code,
                request_id=response.headers.get("x-request-id"),
            )
        return [model["id"] for model in (response.json().get("data") or []) if model.get("id")]

    def fetch_image_as_data_uri(self, url):
        """Download an image and re-encode it as a data URI.

        Remote image URLs are not accepted, so an IMAGE_URL part in a DSS prompt has
        to be inlined before the request goes out.
        """
        logger.info("Inlining image for Cerebras: %s", url)
        response = self.session.get(url, timeout=_IMAGE_DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "image/png").split(";")[0].strip()
        encoded = base64.b64encode(response.content).decode("ascii")
        return "data:{};base64,{}".format(content_type, encoded)
