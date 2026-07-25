import { Expand, Globe2, Minimize2, RotateCcw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { locateGlobeGridPoint } from '../data/globeCoordinateUtils'
import { useGlobeControls } from '../hooks/useGlobeControls'
import { useGlobeManifest } from '../hooks/useGlobeManifest'
import { useGlobeSelection } from '../hooks/useGlobeSelection'
import { useGlobeValues } from '../hooks/useGlobeValues'
import type {
  GlobeCoordinates,
  GlobeGridPoint,
  GlobeMode,
  GlobeResearchDefaults,
} from '../types/globe'
import { resolvePublicAssetUrl } from '../utils/resolvePublicAssetUrl'
import { GlobeCanvas } from './GlobeCanvas'
import { GlobeColorLegend } from './GlobeColorLegend'
import { GlobeEmptyState } from './GlobeEmptyState'
import { GlobeErrorState } from './GlobeErrorState'
import { GlobeFilters } from './GlobeFilters'
import { GlobeLoadingOverlay } from './GlobeLoadingOverlay'
import { GlobeParameterPanel } from './GlobeParameterPanel'

type AtmosphereGlobeProps = {
  manifestUrl?: string
  defaultChannel?: string
  defaultTimestamp?: string
  defaultGrid?: string
  availableModes?: GlobeMode[]
  researchDefaults?: GlobeResearchDefaults
}

const defaultManifestUrl = 'data/globe/manifest.json'

export default function AtmosphereGlobe({
  manifestUrl = defaultManifestUrl,
  defaultChannel = 't2m',
  defaultTimestamp,
  defaultGrid = '0p25',
  availableModes = ['original'],
  researchDefaults = {},
}: AtmosphereGlobeProps) {
  const cardRef = useRef<HTMLElement>(null)
  const { manifest, loading: manifestLoading, error: manifestError, reload } =
    useGlobeManifest(manifestUrl)
  const demoEnabled = import.meta.env.VITE_ENABLE_GLOBE_DEMO === 'true'
  const frames = useMemo(
    () =>
      (manifest?.frames ?? []).filter(
        (frame) => frame.source !== 'demo' || demoEnabled,
      ),
    [demoEnabled, manifest?.frames],
  )
  const selection = useGlobeSelection(frames, availableModes, {
    channel: defaultChannel,
    timestamp: defaultTimestamp,
    grid: defaultGrid,
    ...researchDefaults,
  })
  const { autoRotate, setAutoRotate, reducedMotion, resetSignal, resetView } =
    useGlobeControls()
  const [selectedCoordinates, setSelectedCoordinates] = useState<GlobeCoordinates | null>(null)
  const [textureLoading, setTextureLoading] = useState(false)
  const [textureError, setTextureError] = useState<string | null>(null)
  const [textureRevision, setTextureRevision] = useState(0)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const frame = selection.frame
  const valuesState = useGlobeValues(
    selectedCoordinates ? frame?.valuesUrl : undefined,
    frame ? frame.width * frame.height : undefined,
  )
  const selectedPoint = useMemo<GlobeGridPoint | null>(
    () =>
      selectedCoordinates && frame
        ? locateGlobeGridPoint(selectedCoordinates, frame, valuesState.values ?? undefined)
        : null,
    [frame, selectedCoordinates, valuesState.values],
  )
  const textureUrl = frame ? resolvePublicAssetUrl(frame.textureUrl) : undefined
  const firstAvailableFrame =
    frames.find((candidate) => candidate.mode === selection.mode) ?? frames[0]

  useEffect(() => {
    setTextureError(null)
  }, [textureUrl])

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === cardRef.current)
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || !isFullscreen) return
      if (document.fullscreenElement === cardRef.current) {
        void document.exitFullscreen()
      } else {
        setIsFullscreen(false)
      }
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [isFullscreen])

  const toggleFullscreen = async () => {
    if (document.fullscreenElement === cardRef.current) {
      await document.exitFullscreen()
      return
    }
    if (isFullscreen) {
      setIsFullscreen(false)
      return
    }
    if (cardRef.current?.requestFullscreen) {
      try {
        await cardRef.current.requestFullscreen()
        return
      } catch {
        // Fall back to an in-document fullscreen panel when the API is denied.
      }
    }
    setIsFullscreen(true)
  }

  const retryTexture = () => {
    setTextureError(null)
    setTextureRevision((revision) => revision + 1)
  }

  return (
    <section
      ref={cardRef}
      className={`overflow-hidden border border-[#E4E7EC] bg-white shadow-[0_2px_8px_rgba(16,24,40,0.04)] ${
        isFullscreen
          ? 'fixed inset-0 z-[100] overflow-auto rounded-none border-0 bg-[#F7F8FA] p-4'
          : 'rounded-[20px]'
      }`}
    >
      <div className="flex flex-col gap-3 px-4 py-4 sm:flex-row sm:items-start sm:justify-between sm:px-5">
        <div className="flex min-w-0 items-start gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
            <Globe2 className="h-4.5 w-4.5" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-[18px] font-semibold leading-6 text-[#101828] sm:text-[19px]">
                Глобальное состояние атмосферы ERA5
              </h2>
              {frame?.source === 'demo' ? (
                <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-700">
                  Демо-данные
                </span>
              ) : null}
            </div>
            <p className="mt-0.5 text-[13px] leading-5 text-[#667085]">
              {getLayerDescription(selection.mode)}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center justify-end gap-2 sm:justify-start">
          <IconButton label="Сбросить вид" title="Вернуть вид на Европу и Африку" onClick={resetView}>
            <RotateCcw className="h-4 w-4" aria-hidden="true" />
          </IconButton>
          <IconButton
            label={isFullscreen ? 'Выйти из полноэкранного режима' : 'Во весь экран'}
            title={isFullscreen ? 'Выйти из полноэкранного режима' : 'Во весь экран'}
            onClick={() => void toggleFullscreen()}
          >
            {isFullscreen ? (
              <Minimize2 className="h-4 w-4" aria-hidden="true" />
            ) : (
              <Expand className="h-4 w-4" aria-hidden="true" />
            )}
          </IconButton>
        </div>
      </div>

      <GlobeFilters
        selection={selection}
        availableModes={availableModes}
        autoRotate={autoRotate}
        onAutoRotateChange={setAutoRotate}
      />

      <div className="grid gap-3 p-2 sm:gap-5 sm:p-5 xl:grid-cols-[minmax(0,1fr)_300px]">
        <div className="relative min-w-0 overflow-hidden rounded-2xl border border-slate-200 bg-[radial-gradient(circle_at_50%_45%,#F8FBFF_0%,#EFF6FF_48%,#F8FAFC_100%)]">
          <GlobeCanvas
            textureUrl={textureUrl}
            textureRevision={textureRevision}
            autoRotate={autoRotate}
            isFullscreen={isFullscreen}
            resetSignal={resetSignal}
            onPointSelect={setSelectedCoordinates}
            onTextureLoadingChange={setTextureLoading}
            onTextureError={setTextureError}
          />

          {manifestLoading ? <GlobeLoadingOverlay message="Загрузка каталога слоёв…" /> : null}
          {textureLoading ? <GlobeLoadingOverlay /> : null}
          {manifestError ? (
            <GlobeErrorState
              message="Не удалось загрузить каталог атмосферы"
              details={manifestError}
              onRetry={reload}
            />
          ) : null}
          {textureError ? (
            <GlobeErrorState
              message="Не удалось загрузить слой атмосферы"
              details={textureError}
              onRetry={retryTexture}
            />
          ) : null}
          {!manifestLoading && !manifestError && !textureError && !frame ? (
            <GlobeEmptyState
              message={
                frames.length === 0
                  ? 'Слой ERA5 ещё не подготовлен'
                  : 'Для выбранной комбинации данных нет'
              }
              details={
                frames.length === 0
                  ? demoEnabled
                    ? 'Добавьте записи в public/data/globe/manifest.json.'
                    : 'Добавьте реальные записи в manifest или включите VITE_ENABLE_GLOBE_DEMO=true для разработки.'
                  : `${selection.channel} · ${selection.timestamp || 'время не выбрано'} · ${selection.grid}${
                      selection.level ? ` · ${selection.level} hPa` : ''
                    }`
              }
              onSelectAvailable={
                firstAvailableFrame
                  ? () => selection.selectFrame(firstAvailableFrame)
                  : undefined
              }
            />
          ) : null}
        </div>

        <GlobeParameterPanel
          frame={frame}
          point={selectedPoint}
          valuesLoading={valuesState.loading}
          valuesError={valuesState.error}
        />
      </div>

      {reducedMotion ? (
        <p className="sr-only">
          Автовращение отключено в соответствии с настройкой уменьшения движения.
        </p>
      ) : null}
      <GlobeColorLegend frame={frame} mode={selection.mode} />
    </section>
  )
}

function getLayerDescription(mode: GlobeMode) {
  if (mode === 'reconstruction') {
    return 'Восстановленное моделью поле в выбранный момент времени'
  }
  if (mode === 'absolute-error') {
    return 'Абсолютная разница между ERA5 и восстановлением модели'
  }
  return 'Исходные данные реанализа в выбранный момент времени'
}

function IconButton({
  label,
  title,
  onClick,
  children,
}: {
  label: string
  title: string
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={title}
      onClick={onClick}
      className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-slate-300 bg-white text-slate-600 shadow-sm transition-colors hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
    >
      {children}
    </button>
  )
}
