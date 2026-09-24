"""Public value objects and backend protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str
    chunk_id: str = ""


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Chunk:
    start: float
    end: float
    depth: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def identifier(self) -> str:
        return f"{round(self.start * 1000):010d}_{round(self.end * 1000):010d}"


@dataclass
class ChunkResult:
    chunk: Chunk
    raw_text: str
    segments: list[Segment]
    prompt_tokens: int
    generated_tokens: int
    max_new_tokens: int
    elapsed_seconds: float


@dataclass(frozen=True)
class Validation:
    complete: bool
    terminal_timestamp: bool
    hit_token_limit: bool
    monotonic: bool
    last_segment_end: float
    last_active_audio: float
    tail_gap_seconds: float
    reasons: tuple[str, ...]


class Transcriber(Protocol):
    """Inference boundary used by the pipeline and ASR service adapters."""

    def __call__(
        self,
        clip_path: Path,
        chunk: Chunk,
        max_new_tokens: int,
        max_length: int,
    ) -> ChunkResult:
        ...


class SpeakerEmbedder(Protocol):
    """Return one normalized voice vector for original-timeline spans."""

    model_name: str

    def embed(self, audio_path: Path, spans: list[tuple[float, float]]) -> np.ndarray | None:
        ...
