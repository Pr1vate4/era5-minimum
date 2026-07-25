# Frontend ML Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a polished codec workflow, live ERA5 globe entry point, Grafana navigation, and the approved cobalt visual system to the existing dashboard.

**Architecture:** Keep the current React shell and weather-globe feature. Add a self-contained codec feature with a typed HTTP client and stateful workspace page; extend settings only with service URLs; apply the approved visual language through shared semantic CSS classes.

**Tech Stack:** React 18, TypeScript, Vite, Tailwind CSS, Three.js, Lucide, Vitest.

---

### Task 1: Codec contract and settings

**Files:**
- Create: `src/features/codec/types.ts`
- Create: `src/features/codec/api.ts`
- Create: `src/features/codec/api.test.ts`
- Modify: `src/app/settings.ts`
- Modify: `package.json`
- Modify: `package-lock.json`

- [ ] **Step 1: Add failing tests**

```ts
import { describe, expect, it } from 'vitest'
import { parseCodecStatus, parseCodecJob } from './api'

describe('codec API parsing', () => {
  it('rejects a ready status without a checkpoint', () => {
    expect(() => parseCodecStatus({ ready: true })).toThrow('checkpoint')
  })

  it('keeps serialized and tensor compression ratios separate', () => {
    const job = parseCodecJob({
      id: 'job-1',
      status: 'completed',
      progress: 1,
      metrics: {
        serialized_compression_ratio: 33.2,
        tensor_compression_ratio: 64,
        bitstream_bytes: 1024,
        exact_roundtrip: true,
      },
    })
    expect(job.metrics?.serializedCompressionRatio).toBe(33.2)
    expect(job.metrics?.tensorCompressionRatio).toBe(64)
  })
})
```

- [ ] **Step 2: Run `npm test -- --run src/features/codec/api.test.ts`**

Expected: FAIL because the codec API module does not exist.

- [ ] **Step 3: Implement typed parsers and API client**

```ts
export type CodecJobStatus = 'queued' | 'running' | 'completed' | 'failed'

export function parseCodecStatus(input: unknown): CodecServiceStatus
export function parseCodecJob(input: unknown): CodecJob
export function createCodecClient(options: CodecClientOptions): CodecClient
```

The client calls `GET /api/v1/codec/status`, `POST /api/v1/codec/jobs`, and `GET /api/v1/codec/jobs/{id}` with timeout and abort support. Add normalized `codecBaseUrl` and `grafanaUrl` settings using `VITE_CODEC_API_BASE_URL` and `VITE_GRAFANA_URL`.

- [ ] **Step 4: Run codec tests**

Run: `npm test -- --run src/features/codec/api.test.ts`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add package.json package-lock.json src/app/settings.ts src/features/codec
git commit -m "feat: add typed codec API contract"
```

### Task 2: Codec workspace and navigation

**Files:**
- Create: `src/features/codec/useCodecWorkspace.ts`
- Create: `src/features/codec/CodecDropzone.tsx`
- Create: `src/features/codec/CodecResults.tsx`
- Create: `src/pages/CodecPage.tsx`
- Modify: `src/App.tsx`
- Modify: `src/app/navigationConfig.ts`
- Modify: `src/layout/TopHeader.tsx`

- [ ] **Step 1: Add a failing API-state test**

```ts
it('maps an unavailable service to an explicit disabled state', () => {
  const status = parseCodecStatus({
    ready: false,
    checkpoint: null,
    message: 'Checkpoint is not installed',
  })
  expect(status.ready).toBe(false)
  expect(status.message).toContain('Checkpoint')
})
```

- [ ] **Step 2: Run the focused test**

Run: `npm test -- --run src/features/codec/api.test.ts`

Expected: FAIL until unavailable-state parsing is implemented.

- [ ] **Step 3: Implement the workspace**

The page provides:

```tsx
<CodecDropzone file={file} onFileChange={setFile} />
<RatioSelector value={targetRatio} options={[32, 64]} />
<CodecRunButton disabled={!service.ready || !file} />
<CodecResults job={completedJob} />
```

Accepted files are `.npz`, `.npy`, `.nc`, and `.zarr.zip`; the UI presents idle, validation, upload, queued, running, completed, unavailable, and failed states. It polls only while queued/running and cancels on unmount. Add `/codec` to routing/navigation and a top-header Grafana link that opens `settings.services.grafanaUrl` in a new tab.

- [ ] **Step 4: Verify**

Run: `npm test -- --run src/features/codec/api.test.ts && npm run typecheck`

Expected: PASS with no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/app/navigationConfig.ts src/features/codec src/layout/TopHeader.tsx src/pages/CodecPage.tsx
git commit -m "feat: add ERA5 codec workspace"
```

### Task 3: Visual system and globe presentation

**Files:**
- Modify: `src/index.css`
- Modify: `src/layout/AppShell.tsx`
- Modify: `src/components/Sidebar.tsx`
- Modify: `src/layout/SidebarItem.tsx`
- Modify: `src/components/common/ContentCard.tsx`
- Modify: `src/components/common/PageHeader.tsx`
- Modify: `src/pages/OverviewPage.tsx`
- Modify: `src/features/globe/components/AtmosphereGlobe.tsx`

- [ ] **Step 1: Add semantic visual contracts**

Use `.ui-action`, `.ui-action-primary`, `.ui-panel`, and `.app-backdrop` rather than duplicating button declarations. Primary buttons must have `min-height: 46px`, `border-bottom-width: 4px`, `border-radius: 15px`, and pressed translation.

- [ ] **Step 2: Apply the approved visual language**

Update light tokens to warm off-white/cobalt/charcoal, retain semantic dark tokens, strengthen card grouping, improve the header/sidebar, and make the globe the overview hero. Existing globe auto-rotation and real `/api/v1/layers` loading remain unchanged.

- [ ] **Step 3: Verify responsive and production output**

Run: `npm run typecheck && npm run build`

Expected: TypeScript and Vite build exit 0.

- [ ] **Step 4: Commit**

```bash
git add src
git commit -m "style: refresh dashboard visual system"
```

### Task 4: Final verification

**Files:**
- Verify all modified frontend files.

- [ ] **Step 1: Run tests and build**

Run: `npm test -- --run && npm run typecheck && npm run build`

Expected: all tests pass and production bundle builds.

- [ ] **Step 2: Inspect the dev server**

Run: `npm run dev -- --host 127.0.0.1`

Expected: Vite serves the updated dashboard and `/codec` renders through the hash route.

- [ ] **Step 3: Confirm branch history**

Run: `git status --short && git log --oneline -5`

Expected: clean feature branch with design, API/workspace, and visual commits.
