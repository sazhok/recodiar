"""Stable JSON-adjacent text and WebVTT renderers."""

from __future__ import annotations

import html
from collections.abc import Iterable

from .models import Segment


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def render_vtt(segments: Iterable[Segment], duration: float) -> str:
    lines = [
        "WEBVTT",
        "",
        "NOTE Chunked MOSS transcription; speaker labels are estimates.",
        "",
    ]
    for segment in segments:
        start = max(0.0, segment.start)
        end = min(duration, segment.end)
        if end <= start or not segment.text.strip():
            continue
        digits = "".join(character for character in segment.speaker if character.isdigit())
        speaker_id = f"{int(digits):02d}" if digits else "00"
        lines.extend(
            [
                f"{timestamp(start)} --> {timestamp(end)}",
                f"<speaker id={speaker_id}> {html.escape(segment.text.strip(), quote=False)}",
                "",
            ]
        )
    return "\n".join(lines)


def render_compact_transcript(segments: Iterable[Segment]) -> str:
    return "".join(
        f"[{segment.start:.3f}][{segment.speaker}]{segment.text}[{segment.end:.3f}]"
        for segment in segments
    )
