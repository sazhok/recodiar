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


class VoiceByTime:
    """Answers with a fixed voice per time range, and records what it was asked."""

    model_name = "fake"

    def __init__(self, voices: dict[tuple[float, float], list[float]]) -> None:
        self.voices = voices
        self.calls: list[list[tuple[float, float]]] = []

    def embed(self, audio_path, spans):
        self.calls.append(list(spans))
        for (start, end), vector in self.voices.items():
            if any(start <= s and e <= end for s, e in spans):
                return np.asarray(vector)
        return None


def _two_speakers_then_one_in_overlap():
    left_chunk, right_chunk = Chunk(0.0, 100.0), Chunk(80.0, 180.0)
    left = [
        Segment(10.0, 20.0, "A", "operator greets"),
        Segment(30.0, 40.0, "B", "client explains"),
        Segment(82.0, 88.0, "A", "same overlap words"),
    ]
    right = [
        Segment(82.1, 88.1, "X", "same overlap words"),
        Segment(120.0, 130.0, "Y", "client again"),
    ]
    return [(left_chunk, left), (right_chunk, right)]


def test_embedder_maps_a_voice_the_overlap_did_not_contain() -> None:
    # B spoke in the first chunk but not in the overlap, so the overlap cannot say who Y is.
    embedder = VoiceByTime({(30.0, 40.0): [0.0, 1.0], (120.0, 130.0): [0.0, 1.0],
                            (10.0, 20.0): [1.0, 0.0], (82.0, 88.2): [1.0, 0.0]})
    stats: dict = {}
    mapped = map_speakers(_two_speakers_then_one_in_overlap(), audio_path=None,  # type: ignore[arg-type]
                          embedder=embedder, stats=stats)
    right = {segment.text: segment.speaker for segment in mapped[1][1]}
    assert right == {"same overlap words": "S01", "client again": "S02"}
    assert stats["embedding_chunks"] == 1
    # Only the missing voice and the unmapped label were embedded - not the matched ones.
    assert sorted(embedder.calls) == [[(30.0, 40.0)], [(120.0, 130.0)]]


def test_without_a_match_the_missing_voice_stays_a_new_speaker() -> None:
    embedder = VoiceByTime({(30.0, 40.0): [0.0, 1.0], (120.0, 130.0): [1.0, 0.0]})
    mapped = map_speakers(_two_speakers_then_one_in_overlap(), audio_path=None,  # type: ignore[arg-type]
                          embedder=embedder)
    assert {segment.speaker for segment in mapped[1][1]} == {"S01", "S03"}


def test_embedder_is_not_consulted_when_every_previous_voice_is_matched() -> None:
    left_chunk, right_chunk = Chunk(0.0, 100.0), Chunk(80.0, 180.0)
    left = [
        Segment(82.0, 88.0, "A", "operator in overlap"),
        Segment(90.0, 96.0, "B", "client in overlap"),
    ]
    right = [
        Segment(82.1, 88.1, "X", "operator in overlap"),
        Segment(90.1, 96.1, "Y", "client in overlap"),
        # A third person joins (a speakerphone). Their voice happens to resemble the client's,
        # which is exactly why they must not be embedded against the client.
        Segment(140.0, 150.0, "Z", "somebody new"),
    ]
    embedder = VoiceByTime({(0.0, 200.0): [1.0, 0.0]})
    stats: dict = {}
    mapped = map_speakers([(left_chunk, left), (right_chunk, right)], audio_path=None,  # type: ignore[arg-type]
                          embedder=embedder, stats=stats)
    assert {segment.text: segment.speaker for segment in mapped[1][1]} == {
        "operator in overlap": "S01",
        "client in overlap": "S02",
        "somebody new": "S03",
    }
    assert embedder.calls == [] and stats["embedding_chunks"] == 0
