import numpy as np

from recodiar.io import last_active_audio_time
from recodiar.models import Chunk, ChunkResult, Segment
from recodiar.planning import can_split_chunk, plan_chunks, split_chunk
from recodiar.validation import validate_chunk


def test_plan_chunks_adds_context_overlap() -> None:
    assert plan_chunks(0.0, 1_400.0, 600.0, 45.0) == [
        Chunk(0.0, 600.0),
        Chunk(555.0, 1155.0),
        Chunk(1110.0, 1400.0),
    ]


def test_short_enough_recording_is_one_chunk() -> None:
    assert plan_chunks(0.0, 1_080.0, 600.0, 45.0, whole_max_seconds=1_080.0) == [
        Chunk(0.0, 1_080.0)
    ]
    # One second past the limit is chunked as before.
    assert plan_chunks(0.0, 1_081.0, 600.0, 45.0, whole_max_seconds=1_080.0) == [
        Chunk(0.0, 600.0),
        Chunk(555.0, 1_081.0),
    ]
    # A tail repair measures the span it decodes, not the recording.
    assert plan_chunks(900.0, 1_900.0, 600.0, 45.0, whole_max_seconds=1_080.0) == [
        Chunk(900.0, 1_900.0)
    ]


def test_whole_limit_must_lie_between_one_and_two_chunks() -> None:
    import pytest

    from recodiar.pipeline import PipelineConfig

    PipelineConfig(chunk_seconds=600.0, whole_max_seconds=1_080.0)
    for bad in (599.0, 1_201.0):
        with pytest.raises(ValueError):
            PipelineConfig(chunk_seconds=600.0, whole_max_seconds=bad)


def test_split_respects_actual_child_duration() -> None:
    left, right = split_chunk(Chunk(100.0, 700.0), 40.0)
    assert left == Chunk(100.0, 420.0, 1)
    assert right == Chunk(380.0, 700.0, 1)
    assert can_split_chunk(Chunk(0.0, 40.0), 45.0, 30.0)
    assert not can_split_chunk(Chunk(0.0, 39.0), 45.0, 30.0)


def test_validate_chunk_detects_early_eos_and_audio_gap() -> None:
    result = ChunkResult(
        chunk=Chunk(0.0, 600.0),
        raw_text="[0.1][S01]unfinished",
        segments=[Segment(0.1, 310.0, "S01", "unfinished")],
        prompt_tokens=100,
        generated_tokens=2_000,
        max_new_tokens=16_384,
        elapsed_seconds=1.0,
    )
    validation = validate_chunk(result, last_active=598.0, tail_tolerance=5.0)
    assert not validation.complete
    assert validation.reasons == (
        "missing_terminal_timestamp",
        "active_audio_tail_uncovered",
    )


def test_validate_chunk_accepts_trailing_silence() -> None:
    result = ChunkResult(
        chunk=Chunk(0.0, 600.0),
        raw_text="[0.1][S01]done[570.0]",
        segments=[Segment(0.1, 570.0, "S01", "done")],
        prompt_tokens=100,
        generated_tokens=2_000,
        max_new_tokens=16_384,
        elapsed_seconds=1.0,
    )
    assert validate_chunk(result, last_active=570.0, tail_tolerance=5.0).complete


def test_energy_guard_requires_sustained_activity() -> None:
    samples = np.zeros(1_000, dtype=np.float32)
    samples[400:700] = 0.1
    assert 0.69 <= last_active_audio_time(samples, 1_000, -45.0, 0.01) <= 0.70
    spike = np.zeros(1_000, dtype=np.float32)
    spike[800:850] = 0.1
    assert last_active_audio_time(
        spike,
        1_000,
        -40.0,
        frame_seconds=0.01,
        minimum_active_seconds=0.1,
    ) == 0.0
