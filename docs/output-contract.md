# Output Contract

For an input named `meeting.flac`, the pipeline writes `OUTPUT/meeting/`.

## Final artifacts

- `segments.json`: ordered array of `{start, end, speaker, text, chunk_id}`.
  Times are seconds on the original recording timeline. Speaker IDs are
  anonymous and local to the recording.
- `merged_transcript.txt`: compact MOSS-compatible
  `[start][speaker]text[end]` representation.
- `transcript.vtt`: UTF-8 WebVTT with `<speaker id=NN>` cue text.
- `metadata.json`: duration, model revision, initial plan, successful attempts,
  speaker reconciliation method, how many chunks needed voice embeddings
  (`embedding_chunks`), output counts, and final tail coverage.

## Resumable cache

`chunks/START_MS_END_MS/` contains `raw_transcript.txt`, `segments.json`, and
`metadata.json`. A successful chunk is immutable and reused on subsequent runs.
An incomplete cached result may be revalidated when VAD settings change; if it
remains incomplete, the runner resumes its deterministic child chunks.

CLI progress uses one JSON object per line. Current event names are
`chunk_done`, `chunk_cached`, `skip`, `done`, and `error`. Transcript text is
intentionally excluded from progress events and metadata.
