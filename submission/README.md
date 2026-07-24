# Submission layout

The final package is assembled from verified bind-mounted artifacts; this
repository intentionally does not include placeholder checkpoints, bitstreams,
results, or resource logs.

```text
submission/
├── code/
├── container/
├── checkpoints/
├── bitstreams/
├── manifests/
├── metrics/
├── resources/
└── plots/
```

Use paths relative to `/workspace` inside containers and CLI arguments or
environment variables outside them. Do not use developer-specific absolute
paths.
