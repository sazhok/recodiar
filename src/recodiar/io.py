"""Audio and atomic file I/O helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile

from .models import Chunk


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def read_clip(audio_path: Path, chunk: Chunk) -> tuple[np.ndarray, int]:
    info = soundfile.info(str(audio_path))
    samples, sample_rate = soundfile.read(
        str(audio_path),
        start=round(chunk.start * info.samplerate),
        stop=round(chunk.end * info.samplerate),
        dtype="float32",
        always_2d=False,
    )
    return samples, sample_rate


def last_active_audio_time(
    samples: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float = -45.0,
    frame_seconds: float = 0.03,
    minimum_active_seconds: float = 0.15,
) -> float:
    """Return the last sustained energy-active frame end relative to a clip."""
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    frame_size = max(1, round(sample_rate * frame_seconds))
    minimum_frames = max(1, math.ceil(minimum_active_seconds / frame_seconds))
    active_run_start: int | None = None
    last = 0.0
    for offset in range(0, len(samples), frame_size):
        frame = samples[offset : offset + frame_size]
        if not len(frame):
            continue
        rms = math.sqrt(float(np.mean(np.square(frame, dtype=np.float64))))
        dbfs = 20.0 * math.log10(max(rms, 1e-10))
        if dbfs < threshold_dbfs:
            active_run_start = None
            continue
        if active_run_start is None:
            active_run_start = offset
        run_frames = (offset - active_run_start) // frame_size + 1
        if run_frames >= minimum_frames:
            last = min(len(samples) / sample_rate, (offset + len(frame)) / sample_rate)
    return last
