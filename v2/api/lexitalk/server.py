"""ASGI entrypoint: ``uvicorn lexitalk.server:app``."""

from .main import create_app

app = create_app()
