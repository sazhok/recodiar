"""Speaker reconciliation across independently decoded chunks."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from .models import Chunk, Segment, SpeakerEmbedder, SpeakerTurn

WORD = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    return " ".join(WORD.findall(text.casefold()))


def segment_similarity(left: Segment, right: Segment) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    temporal = intersection / union if union > 0 else 0.0
    lexical = SequenceMatcher(None, normalize_text(left.text), normalize_text(right.text)).ratio()
    return 0.65 * temporal + 0.35 * lexical


def overlap_speaker_votes(
    known: list[Segment],
    incoming: list[Segment],
    overlap_start: float,
    overlap_end: float,
) -> dict[tuple[str, str], float]:
    votes: dict[tuple[str, str], float] = {}
    for left in known:
        if left.end <= overlap_start or left.start >= overlap_end:
            continue
        for right in incoming:
            if right.end <= overlap_start or right.start >= overlap_end:
                continue
            score = segment_similarity(left, right)
            if score >= 0.25:
                key = (right.speaker, left.speaker)
                votes[key] = votes.get(key, 0.0) + score
    return votes


def reference_speaker_votes(
    incoming: list[Segment], reference_turns: list[SpeakerTurn]
) -> dict[tuple[str, str], float]:
    votes: dict[tuple[str, str], float] = {}
    for segment in incoming:
        for turn in reference_turns:
            if turn.end <= segment.start or turn.start >= segment.end:
                continue
            overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
            if overlap > 0:
                key = (segment.speaker, turn.speaker)
                votes[key] = votes.get(key, 0.0) + overlap
    return votes


def greedy_one_to_one(
    scores: dict[tuple[str, str], float], minimum: float
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used_global: set[str] = set()
    for (local, global_label), score in sorted(
        scores.items(), key=lambda item: item[1], reverse=True
    ):
        if score < minimum or local in mapping or global_label in used_global:
            continue
        mapping[local] = global_label
        used_global.add(global_label)
    return mapping


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else -1.0


def map_speakers(
    chunk_segments: list[tuple[Chunk, list[Segment]]],
    audio_path: Path,
    embedder: SpeakerEmbedder | None = None,
    embedding_threshold: float = 0.62,
    reference_turns: list[SpeakerTurn] | None = None,
    stats: dict[str, Any] | None = None,
) -> list[tuple[Chunk, list[Segment]]]:
    """Give every chunk's local labels one recording-wide set.

    In order: reference diarization, then the replicas both decodes share in the overlap, and
    only then voices. The embedder is consulted **only when the overlap could not have
    settled it** - when some voice of the previous chunk found no shared replica there (it
    was silent in those seconds, or the two decodes disagreed about its words) and a local
    label is still unmapped. When every voice of the previous chunk is matched, an unmapped
    label is somebody new, and embedding it would only risk calling them someone else.

    Prototypes are computed at that moment, from every span the missing voice has had so far,
    rather than for every chunk: the check is rare, and a prototype built from the whole
    history is steadier than one built chunk by chunk.
    """
    mapped_chunks: list[tuple[Chunk, list[Segment]]] = []
    reference_labels = {turn.speaker for turn in reference_turns or []}
    reference_numbers = [
        int(digits)
        for label in reference_labels
        if (digits := "".join(character for character in label if character.isdigit()))
    ]
    next_speaker = max(reference_numbers, default=0) + 1
    if stats is not None:
        stats.setdefault("embedding_chunks", 0)

    for chunk, original in sorted(chunk_segments, key=lambda item: item[0].start):
        local_labels = sorted({segment.speaker for segment in original})
        mapping = greedy_one_to_one(
            reference_speaker_votes(original, reference_turns or []), minimum=0.5
        )
        overlap_scores: dict[tuple[str, str], float] = {}
        previous_voices: set[str] = set()
        if mapped_chunks:
            previous_chunk, previous_segments = mapped_chunks[-1]
            previous_voices = {segment.speaker for segment in previous_segments}
            overlap_start = max(chunk.start, previous_chunk.start)
            overlap_end = min(chunk.end, previous_chunk.end)
            if overlap_end > overlap_start:
                overlap_scores = overlap_speaker_votes(
                    previous_segments, original, overlap_start, overlap_end
                )
        used_global = set(mapping.values())
        for (local, global_label), score in sorted(
            overlap_scores.items(), key=lambda item: item[1], reverse=True
        ):
            if score >= 0.5 and local not in mapping and global_label not in used_global:
                mapping[local] = global_label
                used_global.add(global_label)

        missing = previous_voices - set(mapping.values())
        unmapped = [local for local in local_labels if local not in mapping]
        if embedder is not None and missing and unmapped:
            if stats is not None:
                stats["embedding_chunks"] += 1
            history = [segment for _, rows in mapped_chunks for segment in rows]
            prototypes: dict[str, np.ndarray] = {}
            for global_label in sorted(missing):
                spans = [(s.start, s.end) for s in history if s.speaker == global_label]
                vector = embedder.embed(audio_path, spans)
                if vector is not None:
                    prototypes[global_label] = vector
            embedding_scores: dict[tuple[str, str], float] = {}
            for local in unmapped:
                spans = [(s.start, s.end) for s in original if s.speaker == local]
                vector = embedder.embed(audio_path, spans)
                if vector is None:
                    continue
                for global_label, prototype in prototypes.items():
                    embedding_scores[(local, global_label)] = cosine(vector, prototype)
            for local, global_label in greedy_one_to_one(
                embedding_scores, minimum=embedding_threshold
            ).items():
                if local not in mapping and global_label not in mapping.values():
                    mapping[local] = global_label

        for local in local_labels:
            if local not in mapping:
                mapping[local] = f"S{next_speaker:02d}"
                next_speaker += 1
        mapped = [
            Segment(
                segment.start,
                segment.end,
                mapping[segment.speaker],
                segment.text,
                segment.chunk_id,
            )
            for segment in original
        ]
        mapped_chunks.append((chunk, mapped))
    return mapped_chunks


class PyannoteSpeakerEmbedder:
    """Optional pyannote 3.4 adapter compatible with newer runtime libraries."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        from collections import namedtuple

        import huggingface_hub
        import torch
        import torchaudio

        if not hasattr(torchaudio, "AudioMetaData"):
            torchaudio.AudioMetaData = namedtuple(  # type: ignore[attr-defined]
                "AudioMetaData",
                "sample_rate num_frames num_channels bits_per_sample encoding",
            )
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

        original_download = huggingface_hub.hf_hub_download

        def compatible_download(*args: Any, use_auth_token: Any = None, **kwargs: Any) -> Any:
            if use_auth_token is not None and "token" not in kwargs:
                kwargs["token"] = use_auth_token
            return original_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = compatible_download
        from pyannote.audio.core.task import Problem, Resolution, Specifications

        torch.serialization.add_safe_globals(
            [torch.torch_version.TorchVersion, Problem, Resolution, Specifications]
        )
        from pyannote.audio import Inference, Model

        self.model_name = model_name
        model = Model.from_pretrained(model_name)
        if model is None:
            raise RuntimeError(f"Could not load pyannote model {model_name!r}")
        self.inference = Inference(model, window="whole")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.inference.to(self.device)

    def embed(self, audio_path: Path, spans: list[tuple[float, float]]) -> np.ndarray | None:
        import torch

        vectors: list[np.ndarray] = []
        weights: list[float] = []
        info = soundfile.info(str(audio_path))
        for start, end in sorted(spans, key=lambda pair: pair[1] - pair[0], reverse=True)[:5]:
            if end - start < 0.5:
                continue
            samples, sample_rate = soundfile.read(
                str(audio_path),
                start=round(start * info.samplerate),
                stop=round(end * info.samplerate),
                dtype="float32",
                always_2d=True,
            )
            waveform = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
            vector = np.asarray(
                self.inference({"waveform": waveform, "sample_rate": sample_rate})
            ).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if norm > 0 and np.isfinite(norm):
                vectors.append(vector / norm)
                weights.append(min(10.0, end - start))
        if not vectors:
            return None
        result = np.average(np.stack(vectors), axis=0, weights=np.asarray(weights))
        norm = float(np.linalg.norm(result))
        return result / norm if norm > 0 else None
