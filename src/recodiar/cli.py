"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .backends.moss import MossTranscriber
from .io import atomic_write
from .pipeline import ChunkedTranscriptionPipeline, PipelineConfig, recording_name
from .speakers import PyannoteSpeakerEmbedder


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable chunked long-form MOSS transcription"
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-results", type=Path)
    parser.add_argument("--reference-diarization-dir", type=Path)
    parser.add_argument("--pattern", default="*.flac")
    parser.add_argument("--chunk-seconds", type=float, default=600.0)
    parser.add_argument("--overlap-seconds", type=float, default=45.0)
    parser.add_argument("--min-chunk-seconds", type=float, default=120.0)
    parser.add_argument(
        "--tail-merge-ratio",
        type=float,
        default=0.0,
        help="a remainder shorter than this share of --chunk-seconds joins the previous chunk "
        "(0..1; 0.8 decodes 600 s chunks' recordings under 18 min whole)",
    )
    parser.add_argument("--tail-tolerance", type=float, default=5.0)
    parser.add_argument("--vad-threshold-dbfs", type=float, default=-45.0)
    parser.add_argument("--vad-min-active-seconds", type=float, default=0.15)
    parser.add_argument("--max-new-tokens", type=int, default=16_384)
    parser.add_argument("--max-length", type=int, default=131_072)
    parser.add_argument("--speaker-embedding-model")
    parser.add_argument("--embedding-threshold", type=float, default=0.62)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = sorted(args.input_dir.glob(args.pattern))
    if not inputs:
        raise SystemExit(f"No input recordings matched {args.pattern!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        min_chunk_seconds=args.min_chunk_seconds,
        tail_merge_ratio=args.tail_merge_ratio,
        tail_tolerance=args.tail_tolerance,
        vad_threshold_dbfs=args.vad_threshold_dbfs,
        vad_min_active_seconds=args.vad_min_active_seconds,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        embedding_threshold=args.embedding_threshold,
        model_revision=args.model_revision,
    )
    embedder = (
        PyannoteSpeakerEmbedder(args.speaker_embedding_model, device=args.device)
        if args.speaker_embedding_model
        else None
    )
    pipeline = ChunkedTranscriptionPipeline(
        MossTranscriber(args.model, device=args.device, dtype=args.dtype),
        config=config,
        embedder=embedder,
        event_sink=emit,
    )
    failures = 0
    for audio_path in inputs:
        recording = recording_name(audio_path)
        destination = args.output_dir / recording
        if (destination / "metadata.json").exists() and not args.force:
            emit({"event": "skip", "recording": recording})
            continue
        started = time.time()
        try:
            summary = pipeline.process(
                audio_path,
                args.output_dir,
                base_dir=args.base_results,
                reference_diarization_dir=args.reference_diarization_dir,
            )
            emit(
                {
                    "event": "done",
                    "recording": recording,
                    "segments": summary["segments"],
                    "tail_gap_seconds": summary["tail_gap_seconds"],
                    "elapsed_seconds": round(time.time() - started, 2),
                }
            )
        except Exception as error:
            failures += 1
            atomic_write(
                destination / "error.json",
                json.dumps(
                    {"type": type(error).__name__, "message": str(error)},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            emit(
                {
                    "event": "error",
                    "recording": recording,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
