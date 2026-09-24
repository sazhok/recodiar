"""Official MOSS-Transcribe-Diarize backend adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import Chunk, ChunkResult, Segment


class MossTranscriber:
    """Lazily load one MOSS model and reuse it across chunks."""

    def __init__(self, model: str, device: str = "cuda:0", dtype: str = "bf16") -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.runner: Any | None = None

    def __call__(
        self,
        clip_path: Path,
        chunk: Chunk,
        max_new_tokens: int,
        max_length: int,
    ) -> ChunkResult:
        if self.runner is None:
            try:
                from moss_transcribe_diarize.app.model_runner import ModelRunner
            except ImportError as error:
                raise RuntimeError(
                    "MOSS backend is unavailable; install the 'moss' extra"
                ) from error
            self.runner = ModelRunner(self.model, device=self.device, dtype=self.dtype)
        from moss_transcribe_diarize.subtitle import subtitle_segments_from_transcript

        result = self.runner.transcribe(
            clip_path,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            decoding="greedy",
        )
        parsed = subtitle_segments_from_transcript(result.text, postprocess=False)
        return ChunkResult(
            chunk=chunk,
            raw_text=result.text,
            segments=[
                Segment(
                    float(segment.start),
                    float(segment.end),
                    str(segment.speaker),
                    str(segment.text),
                    chunk.identifier,
                )
                for segment in parsed
            ],
            prompt_tokens=result.prompt_len,
            generated_tokens=result.generated_tokens,
            max_new_tokens=max_new_tokens,
            elapsed_seconds=result.elapsed_sec,
        )
