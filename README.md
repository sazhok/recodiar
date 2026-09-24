# Recodiar

Resumable long-form transcription around
[MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize).
The package prevents a premature EOS from silently truncating an hour-long
recording: it transcribes overlapping contextual chunks, validates each result,
recursively splits incomplete chunks, and merges successful leaves back onto
the original timeline.

## Features

- Configurable chunks and overlap (`600 s` and `45 s` by default).
- Detection of malformed EOS, token-limit truncation, invalid timestamps, and
  uncovered active audio.
- Resumable, deterministic per-chunk cache with atomic metadata writes.
- Adaptive retry down to a configurable minimum chunk size.
- A remainder shorter than `--tail-merge-ratio` of a chunk (0..1, e.g. `0.8`)
  joins the chunk before it, so a short tail never costs a chunk boundary: with
  `600 s` chunks a recording under 18 minutes is decoded whole, and a 27-minute
  one as two chunks instead of three.
- Speaker reconciliation from shared context, externally supplied diarization
  turns, or voice embeddings. Embeddings are consulted only when some voice of
  the previous chunk found no shared replica in the overlap and a label is
  still unmapped; the count is `embedding_chunks` in `metadata.json`.
- JSON, compact MOSS transcript, and speaker-tagged WebVTT output.
- Tail repair mode for an existing incomplete long-form MOSS run.

## Installation

Python 3.10 or newer and an NVIDIA CUDA environment are recommended. Install the
appropriate PyTorch/Torchaudio build for the host first, then install this
project:

```bash
python -m venv .venv
. .venv/bin/activate
pip install torch torchaudio --index-url YOUR_PYTORCH_INDEX
pip install -e ".[moss,speaker,dev]"
```

The `speaker` extra is optional. Without it, speaker labels are reconciled from
matching transcript segments in overlapping chunks.

## Usage

```bash
recodiar input/ output/chunked \
  --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --model-revision 704aa4a9c304e8520be88901e0d1960158ef5b15 \
  --chunk-seconds 600 \
  --overlap-seconds 45 \
  --min-chunk-seconds 120 \
  --tail-merge-ratio 0.8 \
  --max-new-tokens 16384 \
  --speaker-embedding-model pyannote/wespeaker-voxceleb-resnet34-LM
```

To repair only a missing tail, pass a prior MOSS result directory:

```bash
recodiar input/ output/repaired --base-results output/full [OPTIONS]
```

Each recording directory contains `segments.json`, `merged_transcript.txt`,
`transcript.vtt`, `metadata.json`, and the resumable `chunks/` cache. Standard
output is JSON Lines so a scheduler or service can consume progress without
parsing human-oriented logs.

## Python API

```python
from pathlib import Path

from recodiar import ChunkedTranscriptionPipeline, PipelineConfig
from recodiar.backends.moss import MossTranscriber

pipeline = ChunkedTranscriptionPipeline(
    transcriber=MossTranscriber("OpenMOSS-Team/MOSS-Transcribe-Diarize"),
    config=PipelineConfig(),
)
summary = pipeline.process(Path("meeting.flac"), Path("output"))
```

The inference backend is a small protocol, so ASRHub can own model lifecycle
and inject its own adapter rather than loading a second model instance. See
[`docs/asrhub-integration.md`](docs/asrhub-integration.md).

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

Unit tests use synthetic audio and fake inference; model weights and private
recordings are neither downloaded nor committed. No project license has been
selected yet; add one before publishing the repository.
