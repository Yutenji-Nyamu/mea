"""Provider interfaces kept separate from RoboTwin evaluation code."""

from abc import ABC, abstractmethod
from pathlib import Path


class MultimodalProvider(ABC):
    """Minimal text and vision contract used by upper-level agents."""

    @abstractmethod
    def text(self, prompt: str, **kwargs) -> str:
        """Return a text response for a text-only prompt."""

    @abstractmethod
    def vision(self, prompt: str, image_path: str | Path, **kwargs) -> str:
        """Return a text response grounded in a local image."""
