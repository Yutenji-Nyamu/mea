"""Backend-neutral identity for one MEA evaluation run."""

from __future__ import annotations

import uuid
from datetime import datetime


def make_evaluation_id() -> str:
    """Create the stable public evaluation identifier used by every backend."""

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"eval_{timestamp}_{uuid.uuid4().hex[:8]}"


__all__ = ["make_evaluation_id"]
