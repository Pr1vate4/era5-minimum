# Frontend codec API contract

The `/codec` frontend talks only to a real codec service. It does not generate
demo metrics or substitute a missing checkpoint.

Configure the service with:

```bash
VITE_CODEC_API_BASE_URL=http://localhost:8000
VITE_GRAFANA_URL=http://localhost:3000
```

An empty codec base URL means same-origin requests.

## Readiness

`GET /api/v1/codec/status`

```json
{
  "ready": true,
  "model_name": "era5-autoencoder",
  "checkpoint": "checkpoints/model.ckpt",
  "message": "Model is ready"
}
```

`ready: true` requires a non-empty checkpoint identifier. When the real
checkpoint is unavailable, return `ready: false`; the frontend keeps submission
disabled.

## Create job

`POST /api/v1/codec/jobs` uses `multipart/form-data`:

- `file`: `.npz`, `.npy`, `.nc`, or `.zarr.zip`;
- `target_ratio`: `32` or `64`, interpreted only as the tensor target ratio.

The frontend does not apply its short metadata timeout to this upload.

## Poll job

`GET /api/v1/codec/jobs/{job_id}`

Queued or running response:

```json
{
  "id": "job-123",
  "status": "running",
  "progress": 0.45,
  "message": "Decoding",
  "error": null,
  "metrics": null,
  "downloads": null,
  "previews": null
}
```

`status` is one of `queued`, `running`, `completed`, or `failed`; `progress` is
clamped to `[0, 1]`. Temporary polling failures are retried by the frontend.

Completed response:

```json
{
  "id": "job-123",
  "status": "completed",
  "progress": 1,
  "message": "Completed",
  "error": null,
  "metrics": {
    "serialized_compression_ratio": 33.2,
    "tensor_compression_ratio": 64,
    "bitstream_bytes": 1048576,
    "exact_roundtrip": true,
    "encode_seconds": 1.2,
    "decode_seconds": 0.8
  },
  "downloads": {
    "bitstream": "/api/v1/codec/jobs/job-123/bitstream",
    "reconstruction": "/api/v1/codec/jobs/job-123/reconstruction"
  },
  "previews": {
    "original": "/api/v1/codec/jobs/job-123/preview/original.png",
    "reconstruction": "/api/v1/codec/jobs/job-123/preview/reconstruction.png"
  }
}
```

For a completed job, `metrics`, `downloads`, and `previews` are mandatory.
`serialized_compression_ratio` must be calculated from the complete serialized
bitstream byte size. `tensor_compression_ratio` is displayed separately and may
be `null`. Preview URLs must return renderable images; the frontend never
constructs scientific fields from synthetic values.

Failed response:

```json
{
  "id": "job-123",
  "status": "failed",
  "progress": 0.45,
  "message": "Input validation failed",
  "error": "Expected 28 channels in canonical order",
  "metrics": null,
  "downloads": null,
  "previews": null
}
```

URLs may be absolute or relative to `VITE_CODEC_API_BASE_URL`. A service on a
different origin must allow the frontend origin through CORS.
