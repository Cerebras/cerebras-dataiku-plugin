"""Which API key a connection authenticates with, and where it came from.

Pure: reads the connection config dict and returns a key or raises. No network, no
DSS imports, so the branching is testable outside DSS -- see tests/.

A connection runs in one of two credential modes:

- SHARED    one admin-managed key in a plugin preset, used by everyone on the
            connection. Simplest to operate, but all Cerebras usage bills to one
            key and shares its rate limits.
- PER_USER  every DSS user supplies their own key, so Cerebras attributes usage
            and rate limits per user.

Both presets expose the key under `api_key`, so the modes differ only in which
preset to read and which screen to name when it turns up empty.
"""

import collections
import logging

logger = logging.getLogger(__name__)

SHARED = "SHARED"
PER_USER = "PER_USER"

DEFAULT_MODE = SHARED

# Everything mode-specific in one place. Each message names the screen that fixes
# it, because the two modes are fixed by different people -- an admin for the
# shared key, the user themselves for their own.
_Mode = collections.namedtuple("_Mode", "preset_param empty_message")

_MODES = {
    SHARED: _Mode(
        preset_param="shared_key",
        empty_message=(
            "No shared Cerebras API key configured. An administrator needs to add a "
            "preset under Administration > Plugins > Cerebras Inference > Settings > "
            "Shared Cerebras API key, and select it on this connection."
        ),
    ),
    PER_USER: _Mode(
        preset_param="user_key",
        empty_message=(
            "No Cerebras API key configured for your account. Open Profile and "
            "settings > Credentials > Plugins and enter your own Cerebras API key. "
            "Create one at https://cloud.cerebras.ai."
        ),
    ),
}


def _mode_from_config(config):
    """The credential mode this connection runs in.

    Absent or blank means SHARED, the safer default.
    """
    raw = config.get("credential_mode")
    name = raw.strip().upper() if isinstance(raw, str) else ""
    if not name:
        return DEFAULT_MODE
    if name not in _MODES:
        raise ValueError(
            "Unrecognised credential mode '{}'. Set 'Credential mode' on the "
            "connection to either 'Shared key (admin-managed)' or 'Individual key "
            "(per user)'.".format(raw)
        )
    return name


def _usable_key(preset):
    """The key held in `preset`, or "" if there is nothing usable in there.

    Normalising every unusable shape down to "" leaves callers one truthiness check.
    Stripping matters: a pasted key often carries a trailing newline, and
    'Bearer <key>\\n' is a 401.
    """
    if not isinstance(preset, dict):
        return ""
    key = preset.get("api_key")
    return key.strip() if isinstance(key, str) else ""


def api_key_from_config(config):
    """Return the API key this connection should authenticate with.

    Raises ValueError naming the screen that fixes it, rather than letting an empty
    key reach the API and come back as an opaque 401.
    """
    mode_name = _mode_from_config(config)
    mode = _MODES[mode_name]

    api_key = _usable_key(config.get(mode.preset_param))
    if not api_key:
        raise ValueError(mode.empty_message)

    logger.info("Authenticating with a %s Cerebras API key", mode_name.lower())
    return api_key
