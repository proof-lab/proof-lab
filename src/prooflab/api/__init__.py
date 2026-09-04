"""Proof Lab API package exposing REST endpoints and background job processing."""

from prooflab.api.app import app, create_app

__all__ = ["app", "create_app"]
