"""Overlap ownership and duplicate suppression."""

from __future__ import annotations

from .models import Chunk, Segment
from .speakers import segment_similarity


def merge_chunks(chunk_segments: list[tuple[Chunk, list[Segment]]]) -> list[Segment]:
    ordered = sorted(chunk_segments, key=lambda item: (item[0].start, item[0].end))
    kept: list[Segment] = []
    for index, (chunk, segments) in enumerate(ordered):
        lower = chunk.start
        upper = chunk.end
        if index:
            lower = (ordered[index - 1][0].end + chunk.start) / 2.0
        if index + 1 < len(ordered):
            upper = (chunk.end + ordered[index + 1][0].start) / 2.0
        for segment in segments:
            if lower <= (segment.start + segment.end) / 2.0 <= upper:
                kept.append(segment)
    kept.sort(key=lambda segment: (segment.start, segment.end, segment.speaker))
    deduplicated: list[Segment] = []
    for segment in kept:
        duplicate = next(
            (
                prior
                for prior in reversed(deduplicated[-8:])
                if prior.speaker == segment.speaker
                and segment_similarity(prior, segment) >= 0.72
            ),
            None,
        )
        if duplicate is None:
            deduplicated.append(segment)
    return deduplicated
