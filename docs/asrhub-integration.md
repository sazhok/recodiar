# ASRHub Integration

The `sazhok/asrhub` repository was not anonymously readable while this package
was extracted, so this document defines the supported boundary without
inventing ASRHub-internal classes. Once an authenticated checkout is available,
implement its provider interface as a thin adapter around the API below.

## Preferred in-process adapter

ASRHub should own the MOSS model instance and implement the `Transcriber`
protocol from `recodiar.models`. Construct one
`ChunkedTranscriptionPipeline` per worker and reuse it across jobs. This avoids
loading MOSS twice and lets ASRHub control its GPU queue, cancellation, and
model lifecycle.

```python
pipeline = ChunkedTranscriptionPipeline(
    transcriber=AsrHubMossAdapter(existing_model),
    config=PipelineConfig(),
    embedder=optional_embedder,
)
summary = pipeline.process(audio_path, job_output_dir)
```

The adapter receives a temporary PCM WAV plus `Chunk`, `max_new_tokens`, and
`max_length`, and returns `ChunkResult`. It must not shift timestamps: chunk
times are local at this boundary and the pipeline converts them to the original
recording timeline.

## Process boundary

If ASRHub isolates GPU models in a separate worker, invoke `recodiar` as a
job and consume its JSON Lines stdout. Treat a zero exit status plus a final
`metadata.json` as success. Preserve the entire recording directory when
retrying so completed chunks are reused.

The stable data contract is documented in `output-contract.md`. ASRHub should
consume `segments.json` as canonical structured output and use VTT only as a
presentation export. Cancellation must terminate between chunks; a future
integration can add a callback to the queue loop if ASRHub exposes cooperative
cancellation.
