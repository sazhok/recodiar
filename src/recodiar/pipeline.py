"""Resumable adaptive transcription orchestration."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile

from .io import atomic_write, last_active_audio_time, read_clip
from .merge import merge_chunks
from .models import Chunk, ChunkResult, Segment, SpeakerEmbedder, SpeakerTurn, Transcriber
from .planning import can_split_chunk, plan_chunks, split_chunk, tail_patch_start
from .rendering import render_compact_transcript, render_vtt
from .speakers import map_speakers
from .validation import TERMINAL_TIMESTAMP, validate_chunk

EventSink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class PipelineConfig:
    chunk_seconds: float = 600.0
    overlap_seconds: float = 45.0
    min_chunk_seconds: float = 120.0
    tail_tolerance: float = 5.0
    vad_threshold_dbfs: float = -45.0
    vad_min_active_seconds: float = 0.15
    max_new_tokens: int = 16_384
    max_length: int = 131_072
    embedding_threshold: float = 0.62
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if self.chunk_seconds <= self.overlap_seconds or self.overlap_seconds < 0:
            raise ValueError("Require chunk_seconds > overlap_seconds >= 0")
        if self.min_chunk_seconds <= 0:
            raise ValueError("min_chunk_seconds must be positive")
        if self.tail_tolerance < 0:
            raise ValueError("tail_tolerance cannot be negative")
        if self.max_new_tokens <= 0 or self.max_length <= 0:
            raise ValueError("token limits must be positive")


def recording_name(audio_path: Path) -> str:
    name = audio_path.name
    return name.removesuffix(".wav.flac").removesuffix(audio_path.suffix)


def load_base_segments(base_dir: Path | None, recording: str) -> tuple[list[Segment], bool]:
    if base_dir is None:
        return [], False
    directory = base_dir / recording
    segments_path = directory / "segments.json"
    raw_path = directory / "raw_transcript.txt"
    if not segments_path.exists():
        return [], False
    rows = json.loads(segments_path.read_text(encoding="utf-8"))
    segments = [
        Segment(
            float(row["start"]),
            float(row["end"]),
            str(row["speaker"]),
            str(row["text"]),
            "base",
        )
        for row in rows
    ]
    terminal = raw_path.exists() and bool(
        TERMINAL_TIMESTAMP.search(raw_path.read_text(encoding="utf-8"))
    )
    return segments, terminal


def load_reference_turns(
    diarization_dir: Path | None, recording: str
) -> list[SpeakerTurn]:
    if diarization_dir is None:
        return []
    path = diarization_dir / f"{recording}.diarization.json"
    if not path.exists():
        raise FileNotFoundError(f"Reference diarization not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_labels = sorted({str(turn["speaker"]) for turn in payload["turns"]})
    label_map = {
        label: f"S{index:02d}" for index, label in enumerate(source_labels, start=1)
    }
    return [
        SpeakerTurn(
            float(turn["start"]),
            float(turn["end"]),
            label_map[str(turn["speaker"])],
        )
        for turn in payload["turns"]
        if float(turn["end"]) > float(turn["start"])
    ]


def _validation_settings(config: PipelineConfig) -> dict[str, float]:
    return {
        "tail_tolerance": config.tail_tolerance,
        "vad_threshold_dbfs": config.vad_threshold_dbfs,
        "vad_min_active_seconds": config.vad_min_active_seconds,
    }


def _cached_segments(path: Path) -> list[Segment]:
    return [
        Segment(**row) for row in json.loads(path.read_text(encoding="utf-8"))
    ]


def _revalidate_cached(
    audio_path: Path,
    chunk: Chunk,
    destination: Path,
    metadata: dict[str, Any],
    segments: list[Segment],
    config: PipelineConfig,
) -> dict[str, Any]:
    samples, sample_rate = read_clip(audio_path, chunk)
    local_segments = [
        Segment(
            segment.start - chunk.start,
            segment.end - chunk.start,
            segment.speaker,
            segment.text,
            segment.chunk_id,
        )
        for segment in segments
    ]
    cached_result = ChunkResult(
        chunk=chunk,
        raw_text=(destination / "raw_transcript.txt").read_text(encoding="utf-8"),
        segments=local_segments,
        prompt_tokens=int(metadata["prompt_tokens"]),
        generated_tokens=int(metadata["generated_tokens"]),
        max_new_tokens=int(metadata["max_new_tokens"]),
        elapsed_seconds=float(metadata["elapsed_seconds"]),
    )
    validation = validate_chunk(
        cached_result,
        last_active_audio_time(
            samples,
            sample_rate,
            config.vad_threshold_dbfs,
            minimum_active_seconds=config.vad_min_active_seconds,
        ),
        config.tail_tolerance,
    )
    metadata["validation"] = asdict(validation)
    metadata["validation_settings"] = _validation_settings(config)
    atomic_write(
        destination / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    return metadata


def transcribe_adaptive(
    audio_path: Path,
    initial: list[Chunk],
    chunks_dir: Path,
    transcriber: Transcriber,
    config: PipelineConfig,
    event_sink: EventSink | None = None,
) -> tuple[list[tuple[Chunk, list[Segment]]], list[dict[str, Any]]]:
    queue = list(initial)
    successful: list[tuple[Chunk, list[Segment]]] = []
    manifest: list[dict[str, Any]] = []
    expected_settings = _validation_settings(config)

    while queue:
        chunk = queue.pop(0)
        destination = chunks_dir / chunk.identifier
        metadata_path = destination / "metadata.json"
        segments_path = destination / "segments.json"
        if metadata_path.exists() and segments_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            segments = _cached_segments(segments_path)
            if metadata.get("validation_settings") != expected_settings:
                metadata = _revalidate_cached(
                    audio_path, chunk, destination, metadata, segments, config
                )
            validation_data = metadata["validation"]
            if validation_data["complete"]:
                successful.append((chunk, segments))
                manifest.append(metadata)
                if event_sink:
                    event_sink({"event": "chunk_cached", "chunk": chunk.identifier})
                continue
            if can_split_chunk(chunk, config.overlap_seconds, config.min_chunk_seconds):
                queue[0:0] = list(split_chunk(chunk, config.overlap_seconds))
                manifest.append(metadata)
                continue
            raise RuntimeError(
                f"Chunk {chunk.identifier} remains incomplete at minimum size: "
                + ", ".join(validation_data["reasons"])
            )

        destination.mkdir(parents=True, exist_ok=True)
        samples, sample_rate = read_clip(audio_path, chunk)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", dir=chunks_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            soundfile.write(str(temporary_path), samples, sample_rate, subtype="PCM_16")
            result = transcriber(
                temporary_path,
                chunk,
                config.max_new_tokens,
                config.max_length,
            )
        finally:
            temporary_path.unlink(missing_ok=True)

        validation = validate_chunk(
            result,
            last_active_audio_time(
                samples,
                sample_rate,
                config.vad_threshold_dbfs,
                minimum_active_seconds=config.vad_min_active_seconds,
            ),
            config.tail_tolerance,
        )
        global_segments = [
            Segment(
                round(chunk.start + segment.start, 3),
                round(chunk.start + segment.end, 3),
                segment.speaker,
                segment.text,
                chunk.identifier,
            )
            for segment in result.segments
        ]
        metadata = {
            "chunk": asdict(chunk),
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "max_new_tokens": config.max_new_tokens,
            "elapsed_seconds": result.elapsed_seconds,
            "validation": asdict(validation),
            "validation_settings": expected_settings,
        }
        atomic_write(destination / "raw_transcript.txt", result.raw_text + "\n")
        atomic_write(
            segments_path,
            json.dumps(
                [asdict(segment) for segment in global_segments],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        atomic_write(
            metadata_path,
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )
        manifest.append(metadata)
        if event_sink:
            event_sink(
                {
                    "event": "chunk_done",
                    "chunk": chunk.identifier,
                    "complete": validation.complete,
                    "generated_tokens": result.generated_tokens,
                    "reasons": validation.reasons,
                }
            )
        if validation.complete:
            successful.append((chunk, global_segments))
        elif can_split_chunk(chunk, config.overlap_seconds, config.min_chunk_seconds):
            queue[0:0] = list(split_chunk(chunk, config.overlap_seconds))
        else:
            raise RuntimeError(
                f"Chunk {chunk.identifier} remains incomplete at minimum size: "
                + ", ".join(validation.reasons)
            )
    return successful, manifest


def _base_as_chunk(segments: list[Segment]) -> tuple[Chunk, list[Segment]] | None:
    if not segments:
        return None
    return Chunk(0.0, max(segment.end for segment in segments)), segments


class ChunkedTranscriptionPipeline:
    """Integration-friendly long-form transcription facade."""

    def __init__(
        self,
        transcriber: Transcriber,
        config: PipelineConfig | None = None,
        embedder: SpeakerEmbedder | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.transcriber = transcriber
        self.config = config or PipelineConfig()
        self.embedder = embedder
        self.event_sink = event_sink

    def process(
        self,
        audio_path: Path,
        output_dir: Path,
        *,
        base_dir: Path | None = None,
        reference_diarization_dir: Path | None = None,
    ) -> dict[str, Any]:
        info = soundfile.info(str(audio_path))
        recording = recording_name(audio_path)
        destination = output_dir / recording
        base_segments, base_terminal = load_base_segments(base_dir, recording)
        base_last = max((segment.end for segment in base_segments), default=0.0)
        base_complete = (
            base_terminal
            and base_last >= info.duration - self.config.tail_tolerance
        )
        if base_complete:
            initial: list[Chunk] = []
            chunk_results = [(Chunk(0.0, info.duration), base_segments)]
            manifest: list[dict[str, Any]] = []
            patch_start = info.duration
        else:
            patch_start = tail_patch_start(
                base_segments, info.duration, self.config.overlap_seconds
            )
            initial = plan_chunks(
                patch_start,
                info.duration,
                self.config.chunk_seconds,
                self.config.overlap_seconds,
            )
            chunk_results, manifest = transcribe_adaptive(
                audio_path,
                initial,
                destination / "chunks",
                self.transcriber,
                self.config,
                self.event_sink,
            )
            base = _base_as_chunk(base_segments)
            if base is not None:
                chunk_results.insert(0, base)

        reference_turns = load_reference_turns(reference_diarization_dir, recording)
        mapped = map_speakers(
            chunk_results,
            audio_path,
            embedder=self.embedder,
            embedding_threshold=self.config.embedding_threshold,
            reference_turns=reference_turns,
        )
        merged = merge_chunks(mapped)
        payload = [asdict(segment) for segment in merged]
        atomic_write(
            destination / "segments.json",
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
        atomic_write(destination / "transcript.vtt", render_vtt(merged, info.duration))
        atomic_write(
            destination / "merged_transcript.txt",
            render_compact_transcript(merged) + "\n",
        )
        base_used = _base_as_chunk(base_segments) is not None
        final_last = max((segment.end for segment in merged), default=0.0)
        summary = {
            "input": str(audio_path),
            "recording": recording,
            "duration": info.duration,
            "model_revision": self.config.model_revision,
            "base_results": str(base_dir) if base_dir else None,
            "base_complete": base_complete,
            "patch_start": patch_start,
            "config": asdict(self.config),
            "initial_chunks": [asdict(chunk) for chunk in initial],
            "attempts": manifest,
            "successful_leaf_chunks": len(chunk_results) - (1 if base_used else 0),
            "segments": len(merged),
            "speakers": sorted({segment.speaker for segment in merged}),
            "last_segment_end": final_last,
            "tail_gap_seconds": max(0.0, info.duration - final_last),
            "speaker_mapping": (
                "reference_diarization_then_overlap_then_embedding"
                if reference_turns and self.embedder
                else "reference_diarization_then_overlap"
                if reference_turns
                else "overlap_then_pyannote_embedding"
                if self.embedder
                else "overlap"
            ),
            "speaker_embedding_model": getattr(self.embedder, "model_name", None),
        }
        atomic_write(
            destination / "metadata.json",
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
        (destination / "error.json").unlink(missing_ok=True)
        return summary
