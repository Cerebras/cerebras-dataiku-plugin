"""Shared helpers for the Cerebras LLM Mesh plugin.

`mapping` holds the pure translation between the Dataiku LLM contract and the
Cerebras chat-completions payloads (no I/O, unit-testable). `client` holds the
HTTP layer.
"""

DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"
