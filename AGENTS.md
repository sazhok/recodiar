# Repository Guidelines

## Project Structure

Production code lives under `src/recodiar/`. Keep the MOSS-specific adapter
in `backends/`; the chunk planner, validation, speaker reconciliation, merging,
and rendering layers must remain backend-independent. Tests belong in `tests/`
and should use synthetic audio or fake inference. Integration notes and stable
output contracts live in `docs/`. Never place model weights, source recordings,
or generated transcripts in this repository.

## Development Commands

- `pip install -e ".[dev]"` installs the lightweight core and developer tools.
- `pip install -e ".[moss,speaker,dev]"` adds the MOSS and pyannote adapters;
  install a host-appropriate PyTorch build first.
- `pytest` runs the offline unit suite without downloading models.
- `ruff check .` checks imports, style, and common Python errors.
- `python -m build` creates source and wheel distributions when `build` is
  installed.
- `git diff --check` detects whitespace errors before review.

## Coding and Testing Conventions

Use Python 3.10 or newer, four-space indentation, complete type hints, `snake_case` for
functions/modules, and `PascalCase` for classes. Keep optional ML imports lazy so
the core test suite remains CPU-only. Use dataclasses for public value objects
and protocols for model or embedding backends. Tests are named
`test_<behavior>.py`; cover timestamp boundaries, premature EOS, token limits,
adaptive splits, cache resumption, speaker mapping, and overlap deduplication.

## Outputs, Security, and Reviews

All generated outputs must target a caller-provided directory and use atomic
writes. Logs and progress events must not contain transcript text. Never commit
Hugging Face tokens, `.env` files, audio, model caches, or transcript excerpts.
Use short imperative commit subjects such as `Add adaptive chunk validation`.
Pull requests should document contract changes, tests run, GPU/runtime versions
used for integration tests, and known transcription or diarization limitations.
