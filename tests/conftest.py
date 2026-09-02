"""Shared pytest fixtures for the Proof Lab test suite."""

import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Clear the get_settings lru_cache before and after every test.

    This guarantees that monkeypatched environment variables are picked
    up by a fresh settings load and that one test cannot bleed state
    into the next through the cache.
    """
    from prooflab.config import get_settings

    get_settings.cache_clear()
    yield  # type: ignore[misc]
    get_settings.cache_clear()
