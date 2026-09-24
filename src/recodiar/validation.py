"""Content-free completeness checks for generated chunks."""

from __future__ import annotations

import re

from .models import ChunkResult, Validation

TERMINAL_TIMESTAMP = re.compile(r"\[\d+(?:\.\d+)?\]\s*$")


def validate_chunk(
    result: ChunkResult,
    last_active: float,
    tail_tolerance: float,
) -> Validation:
    terminal = bool(TERMINAL_TIMESTAMP.search(result.raw_text))
    hit_limit = result.generated_tokens >= result.max_new_tokens
    starts = [segment.start for segment in result.segments]
    intervals_valid = all(
        0 <= segment.start < segment.end <= result.chunk.duration + tail_tolerance
        for segment in result.segments
    )
    monotonic = intervals_valid and all(
        starts[index] <= starts[index + 1] for index in range(len(starts) - 1)
    )
    last_end = max((segment.end for segment in result.segments), default=0.0)
    gap = max(0.0, last_active - last_end)
    reasons: list[str] = []
    if not terminal:
        reasons.append("missing_terminal_timestamp")
    if hit_limit:
        reasons.append("token_limit")
    if not monotonic:
        reasons.append("invalid_or_nonmonotonic_segments")
    if gap > tail_tolerance:
        reasons.append("active_audio_tail_uncovered")
    if not result.segments and last_active > tail_tolerance:
        reasons.append("no_segments_for_active_audio")
    return Validation(
        complete=not reasons,
        terminal_timestamp=terminal,
        hit_token_limit=hit_limit,
        monotonic=monotonic,
        last_segment_end=last_end,
        last_active_audio=last_active,
        tail_gap_seconds=gap,
        reasons=tuple(reasons),
    )
