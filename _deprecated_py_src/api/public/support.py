"""Python-hosted public support route definitions."""

from fastapi import APIRouter

from .catalog import python_host_router as catalog_python_host_router


def build_python_public_support_router() -> APIRouter:
    """Return public routes that still belong to the Python host."""
    support_router = APIRouter()

    support_router.include_router(catalog_python_host_router)
    return support_router


python_public_support_router = build_python_public_support_router()

__all__ = ["build_python_public_support_router", "python_public_support_router"]
