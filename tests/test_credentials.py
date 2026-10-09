"""Tests for API key resolution across the two credential modes.

This branching decides whether a request is authenticated at all, and the two modes
are fixed by different people -- so the error messages are asserted, not just the
exception type.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-lib"))

from dku_cerebras.credentials import PER_USER, SHARED, api_key_from_config  # noqa: E402


class TestSharedMode:
    def test_reads_the_admin_managed_preset(self):
        config = {"credential_mode": SHARED, "shared_key": {"api_key": "csk-shared"}}
        assert api_key_from_config(config) == "csk-shared"

    def test_ignores_a_user_preset_that_happens_to_be_filled(self):
        config = {
            "credential_mode": SHARED,
            "shared_key": {"api_key": "csk-shared"},
            "user_key": {"api_key": "csk-mine"},
        }
        assert api_key_from_config(config) == "csk-shared"

    def test_an_empty_preset_points_the_admin_at_the_right_screen(self):
        config = {"credential_mode": SHARED, "shared_key": {}}
        with pytest.raises(ValueError, match="Administration > Plugins"):
            api_key_from_config(config)


class TestPerUserMode:
    def test_reads_the_individual_credential(self):
        config = {"credential_mode": PER_USER, "user_key": {"api_key": "csk-mine"}}
        assert api_key_from_config(config) == "csk-mine"

    def test_ignores_a_shared_preset_that_happens_to_be_filled(self):
        config = {
            "credential_mode": PER_USER,
            "shared_key": {"api_key": "csk-shared"},
            "user_key": {"api_key": "csk-mine"},
        }
        assert api_key_from_config(config) == "csk-mine"

    def test_an_unfilled_credential_points_the_user_at_their_own_settings(self):
        """The admin creates the preset empty; until the user fills it in, this fires."""
        config = {"credential_mode": PER_USER, "user_key": {"api_key": ""}}
        with pytest.raises(ValueError, match="Profile and settings > Credentials"):
            api_key_from_config(config)

    def test_a_preset_the_user_never_opened_is_not_a_crash(self):
        config = {"credential_mode": PER_USER}
        with pytest.raises(ValueError, match="Profile and settings > Credentials"):
            api_key_from_config(config)


class TestModeHandling:
    def test_a_missing_mode_defaults_to_shared(self):
        """A connection saved before the mode existed carries only the shared preset."""
        assert api_key_from_config({"shared_key": {"api_key": "csk-shared"}}) == "csk-shared"

    def test_a_blank_mode_defaults_to_shared(self):
        config = {"credential_mode": "  ", "shared_key": {"api_key": "csk-shared"}}
        assert api_key_from_config(config) == "csk-shared"

    def test_a_non_string_mode_defaults_to_shared(self):
        config = {"credential_mode": None, "shared_key": {"api_key": "csk-shared"}}
        assert api_key_from_config(config) == "csk-shared"

    def test_the_mode_is_matched_case_insensitively(self):
        config = {"credential_mode": "per_user", "user_key": {"api_key": "csk-mine"}}
        assert api_key_from_config(config) == "csk-mine"

    def test_an_unrecognised_mode_is_rejected_rather_than_defaulted(self):
        """Silently falling back would authenticate as somebody the admin did not pick."""
        with pytest.raises(ValueError, match="Unrecognised credential mode 'SSO'"):
            api_key_from_config({"credential_mode": "SSO"})


class TestKeyHygiene:
    def test_surrounding_whitespace_is_stripped(self):
        """A pasted key often carries a trailing newline; Bearer <key>\\n is a 401."""
        assert api_key_from_config({"shared_key": {"api_key": "  csk-padded\n"}}) == "csk-padded"

    def test_a_whitespace_only_key_counts_as_missing(self):
        with pytest.raises(ValueError, match="No shared Cerebras API key"):
            api_key_from_config({"shared_key": {"api_key": "   "}})

    def test_a_non_string_key_counts_as_missing(self):
        with pytest.raises(ValueError, match="No shared Cerebras API key"):
            api_key_from_config({"shared_key": {"api_key": 12345}})

    def test_a_preset_that_is_not_a_dict_counts_as_missing(self):
        with pytest.raises(ValueError, match="No shared Cerebras API key"):
            api_key_from_config({"shared_key": "csk-not-a-preset"})
