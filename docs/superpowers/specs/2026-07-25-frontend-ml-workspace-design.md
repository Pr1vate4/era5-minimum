# ERA5 Frontend ML Workspace Design

## Goal

Turn the existing research dashboard into a polished operator interface where a user can inspect real ERA5 weather layers on the rotating globe, submit a compatible ERA5 input to the codec, monitor reconstruction, download artifacts, and open Grafana directly.

## Scope

- Preserve the current React/Vite dashboard and Three.js globe.
- Add a dedicated `/codec` workspace for the primary model workflow.
- Restyle the shared shell, cards, controls, and buttons with a warm off-white surface, near-black typography, cobalt accent, soft borders, and tactile four-pixel button edges based on the supplied reference.
- Add a configurable external Grafana link to the top-level navigation.
- Keep existing experiment-analysis routes and metric definitions intact.
- Do not fabricate model output. An unavailable backend or checkpoint is shown explicitly.

## User Flow

1. The overview opens with the rotating weather globe and real layer controls.
2. The user selects a timestamp and one of the canonical weather channels; the layer is loaded through the existing weather API and rendered on the globe.
3. The user opens **Сжатие**, drops a supported ERA5 file, chooses 32× or 64×, and starts processing.
4. The UI validates the file, submits it through a typed codec API client, and shows upload, processing, and completion states.
5. A completed run shows original/reconstruction previews, serialized bitstream size, serialized compression ratio, tensor compression ratio as a separate metric, timings, and exact-roundtrip status.
6. The user downloads the bitstream or reconstructed artifact.
7. The **Grafana** action opens the configured monitoring dashboard in a new tab.

## Frontend Architecture

- `src/features/codec/` owns codec API types, the upload workflow, progress state, metric presentation, and downloads.
- `src/pages/CodecPage.tsx` composes the feature without embedding transport logic.
- Shared buttons and panels remain in `src/components/common/`.
- `src/app/settings.ts` stores `codecApiBaseUrl` and `grafanaUrl`, with Vite environment defaults.
- Existing `ResultsProvider` remains responsible only for experiment result JSON.
- Existing weather-globe hooks remain responsible for `/api/v1/variables`, `/api/v1/timestamps`, and `/api/v1/layers`.

## Codec API Contract

The frontend integration uses:

- `GET /api/v1/codec/status` — service and checkpoint readiness.
- `POST /api/v1/codec/jobs` — multipart submission with `file` and `target_ratio`.
- `GET /api/v1/codec/jobs/{job_id}` — queued/running/completed/failed state and progress.
- Completed job payloads provide metrics plus download URLs for bitstream and reconstruction.

Until the ML implementation exposes this contract, the page remains fully usable for validation and connection diagnostics but disables submission with an explicit readiness message. It never substitutes synthetic metrics.

## Visual System

- Background: warm `#f5f4ef` with a faint cool radial wash.
- Chrome and cards: white/translucent white with neutral gray borders.
- Accent: cobalt blue around `#175cc7`; dark accent for pressed states.
- Text: charcoal rather than blue-gray.
- Primary actions: 46–48 px height, 14–15 px radius, 1 px border with a 4 px bottom edge, short lift on hover, and a pressed translation.
- Cards: larger radii, restrained shadows, clear section grouping, and responsive stacking.
- Dark theme remains supported with equivalent semantic variables.

## Accessibility and Responsive Behaviour

- Minimum 44 px touch targets on narrow screens.
- Keyboard-operable upload, buttons, navigation, and globe controls.
- Visible focus rings and status announcements through `aria-live`.
- Mobile layout converts the sidebar to a compact navigation surface and stacks workflow panels.
- External Grafana navigation includes an accessible label and external-link indicator.

## Error Handling

- Distinguish invalid file, unreachable API, model unavailable, failed job, and expired download.
- Preserve the selected file and settings after retryable transport failures.
- Cancel polling when leaving the page.
- Never claim successful compression without a completed backend payload containing serialized byte counts and exact-roundtrip status.

## Verification

- Unit tests cover settings normalization, codec payload parsing, honest metric labels, and readiness/error states.
- TypeScript typecheck and production build must pass.
- Existing Python changes are avoided unless the backend contract must be implemented in this branch; if Python changes become necessary, `make verify` is mandatory.
- Manual smoke checks cover desktop/mobile layout, globe layer loading, unavailable-model UX, Grafana navigation, and the completed-job presentation using a contract fixture only in tests.
