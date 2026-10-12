"""Model catalog knowledge that the Cerebras API does not expose.

`GET /v1/models` returns only `id` and `owned_by` -- no context window, no capability
flags -- so vision support has to be known locally. The connection has a checkbox to
force it on for anything newer than this list.
"""

# Kept in step with https://inference-docs.cerebras.ai/capabilities/image-inputs.
# kimi-k2.7-code is customer-trials-only, so most keys cannot reach it; listing it
# costs nothing and saves the few who can from ticking the override checkbox.
VISION_MODELS = frozenset(["qwen-3.8-27b", "gemma-4-31b", "kimi-k2.7-code"])


def model_label(model_id):
    """Human label for a model ID, flagging the ones that accept images.

    Used by tests/live_check.py; nothing in the plugin calls this at runtime.
    """
    return "{} (vision)".format(model_id) if model_id in VISION_MODELS else model_id
