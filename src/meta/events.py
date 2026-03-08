"""Training event emitter — routes events to HTTP webhook or local sync."""
from __future__ import annotations

from src.logging.events import Event, emit_event_http, emit_event_sync


class TrainingEventEmitter:
    """Routes training events to the dashboard.

    - If webhook_url is set: POST events via HTTP (Modal → local dashboard).
    - If not set: write to local JSONL via emit_event_sync.
    """

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url

    def emit(self, event: Event):
        if self.webhook_url:
            emit_event_http(event, self.webhook_url)
        else:
            emit_event_sync(event)
