"""Deterministic initial and adaptive chunk planning."""

from __future__ import annotations

from .models import Chunk


def plan_chunks(
    start: float,
    end: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[Chunk]:
    if end <= start:
        return []
    if chunk_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("Require chunk_seconds > overlap_seconds >= 0")
    chunks: list[Chunk] = []
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + chunk_seconds)
        chunks.append(Chunk(round(cursor, 6), round(chunk_end, 6)))
        if chunk_end >= end:
            break
        cursor = chunk_end - overlap_seconds
    return chunks


def split_chunk(chunk: Chunk, overlap_seconds: float) -> tuple[Chunk, Chunk]:
    midpoint = (chunk.start + chunk.end) / 2.0
    half_overlap = min(overlap_seconds / 2.0, chunk.duration / 4.0)
    return (
        Chunk(chunk.start, midpoint + half_overlap, chunk.depth + 1),
        Chunk(midpoint - half_overlap, chunk.end, chunk.depth + 1),
    )


def can_split_chunk(chunk: Chunk, overlap_seconds: float, minimum_seconds: float) -> bool:
    left, right = split_chunk(chunk, overlap_seconds)
    return min(left.duration, right.duration) >= minimum_seconds


def tail_patch_start(base_segments: list, duration: float, overlap: float) -> float:
    if not base_segments:
        return 0.0
    last_end = min(duration, max(segment.end for segment in base_segments))
    return max(0.0, last_end - overlap)
