import json
from pathlib import Path

import numpy as np
import soundfile

from recodiar.models import Chunk, ChunkResult, Segment
from recodiar.pipeline import (
    ChunkedTranscriptionPipeline,
    PipelineConfig,
    transcribe_adaptive,
)


class FakeTranscriber:
    def __init__(self, fail_parent: bool = False) -> None:
        self.fail_parent = fail_parent
        self.calls: list[Chunk] = []

    def __call__(self, clip_path, chunk, max_new_tokens, max_length):
        self.calls.append(chunk)
        complete = not self.fail_parent or chunk.depth > 0
        end = chunk.duration if complete else 1.0
        raw = f"[0.0][S01]speech[{end}]" if complete else "[0.0][S01]speech"
        return ChunkResult(
            chunk=chunk,
            raw_text=raw,
            segments=[Segment(0.001, end, "S01", "speech", chunk.identifier)],
            prompt_tokens=10,
            generated_tokens=20,
            max_new_tokens=max_new_tokens,
            elapsed_seconds=0.1,
        )


def write_audio(path: Path, seconds: float = 4.0) -> None:
    soundfile.write(path, np.full(round(seconds * 1_000), 0.1, dtype=np.float32), 1_000)


def test_adaptive_split_is_resumable(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    write_audio(audio_path)
    backend = FakeTranscriber(fail_parent=True)
    config = PipelineConfig(
        chunk_seconds=4.0,
        overlap_seconds=0.5,
        min_chunk_seconds=2.0,
        tail_tolerance=0.1,
        vad_threshold_dbfs=-45.0,
        max_new_tokens=100,
        max_length=1_000,
    )
    arguments = (audio_path, [Chunk(0.0, 4.0)], tmp_path / "chunks", backend, config)
    successful, manifest = transcribe_adaptive(*arguments)
    assert len(backend.calls) == 3
    assert len(successful) == 2
    assert len(manifest) == 3

    backend.calls.clear()
    resumed, resumed_manifest = transcribe_adaptive(*arguments)
    assert backend.calls == []
    assert len(resumed) == 2
    assert len(resumed_manifest) == 3


def test_pipeline_writes_stable_outputs(tmp_path) -> None:
    audio_path = tmp_path / "meeting.wav"
    output_dir = tmp_path / "output"
    write_audio(audio_path)
    backend = FakeTranscriber()
    pipeline = ChunkedTranscriptionPipeline(
        backend,
        PipelineConfig(
            chunk_seconds=3.0,
            overlap_seconds=1.0,
            min_chunk_seconds=1.0,
            tail_tolerance=0.1,
            max_new_tokens=100,
            max_length=1_000,
            model_revision="test",
        ),
    )
    summary = pipeline.process(audio_path, output_dir)
    destination = output_dir / "meeting"
    assert summary["successful_leaf_chunks"] == 2
    assert summary["tail_gap_seconds"] == 0.0
    assert json.loads((destination / "segments.json").read_text())
    assert "<speaker id=01>" in (destination / "transcript.vtt").read_text()
    assert (destination / "metadata.json").exists()
    assert not (destination / "error.json").exists()
