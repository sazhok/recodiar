import numpy as np

from recodiar.merge import merge_chunks
from recodiar.models import Chunk, Segment, SpeakerTurn
from recodiar.speakers import map_speakers


def test_overlap_maps_local_speaker_and_removes_duplicate() -> None:
    left_chunk = Chunk(0.0, 100.0)
    right_chunk = Chunk(80.0, 180.0)
    left = [
        Segment(82.0, 88.0, "A", "same overlap words", left_chunk.identifier),
        Segment(20.0, 30.0, "A", "earlier", left_chunk.identifier),
    ]
    right = [
        Segment(82.1, 88.1, "X", "same overlap words", right_chunk.identifier),
        Segment(120.0, 130.0, "X", "later", right_chunk.identifier),
    ]
    mapped = map_speakers([(left_chunk, left), (right_chunk, right)], audio_path=None)  # type: ignore[arg-type]
    assert {segment.speaker for _, rows in mapped for segment in rows} == {"S01"}
    assert [segment.text for segment in merge_chunks(mapped)] == [
        "earlier",
        "same overlap words",
        "later",
    ]


def test_voice_embedding_maps_speaker_without_overlap() -> None:
    class SameVoiceEmbedder:
        model_name = "fake"

        def embed(self, audio_path, spans):
            return np.asarray([1.0, 0.0])

    mapped = map_speakers(
        [
            (Chunk(0.0, 50.0), [Segment(10.0, 20.0, "A", "first")]),
            (Chunk(60.0, 110.0), [Segment(70.0, 80.0, "X", "second")]),
        ],
        audio_path=None,  # type: ignore[arg-type]
        embedder=SameVoiceEmbedder(),
    )
    assert [rows[0].speaker for _, rows in mapped] == ["S01", "S01"]


def test_reference_diarization_stabilizes_speakers() -> None:
    reference = [
        SpeakerTurn(10.0, 20.0, "S01"),
        SpeakerTurn(70.0, 80.0, "S01"),
    ]
    mapped = map_speakers(
        [
            (Chunk(0.0, 50.0), [Segment(10.0, 20.0, "A", "first")]),
            (Chunk(60.0, 110.0), [Segment(70.0, 80.0, "X", "second")]),
        ],
        audio_path=None,  # type: ignore[arg-type]
        reference_turns=reference,
    )
    assert [rows[0].speaker for _, rows in mapped] == ["S01", "S01"]
