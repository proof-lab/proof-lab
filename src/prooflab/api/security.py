"""Authentication and security dependencies for protected API endpoints."""

from __future__ import annotations

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
ADMIN_KEY_HEADER = APIKeyHeader(name="X-Admin-Key", auto_error=False)
HTTP_BEARER = HTTPBearer(auto_error=False)

DEFAULT_API_KEY = os.getenv("PROOF_API_KEY", "prooflab-dev-key")
DEFAULT_ADMIN_KEY = os.getenv("PROOF_ADMIN_KEY", "prooflab-admin-key")


async def verify_api_key(
    api_key_header: str | None = Security(API_KEY_HEADER),
    bearer: HTTPAuthorizationCredentials | None = Security(HTTP_BEARER),
) -> str:
    """Verify presence of valid API key or Bearer token for production access."""
    expected = os.getenv("PROOF_API_KEY", DEFAULT_API_KEY)
    admin_key = os.getenv("PROOF_ADMIN_KEY", DEFAULT_ADMIN_KEY)

    token = api_key_header or (bearer.credentials if bearer else None)

    if not token or (token != expected and token != admin_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return token


async def verify_admin_key(
    admin_key_header: str | None = Security(ADMIN_KEY_HEADER),
    api_key_header: str | None = Security(API_KEY_HEADER),
    bearer: HTTPAuthorizationCredentials | None = Security(HTTP_BEARER),
) -> str:
    """Verify administrator authorization for hazardous or high-risk operational gates."""
    expected_admin = os.getenv("PROOF_ADMIN_KEY", DEFAULT_ADMIN_KEY)

    token = admin_key_header or api_key_header or (bearer.credentials if bearer else None)

    if not token or token != expected_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privilege required for this operational gate",
            headers={"WWW-Authenticate": "AdminKey"},
        )
    return token
