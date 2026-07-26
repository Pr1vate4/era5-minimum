import { Expand, Globe2, Minimize2, RotateCcw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  FrameCompressionButton,
  FrameCompressionNotice,
  FrameCompressionReport,
} from '../../codec/CurrentFrameCompression'
import { useCurrentFrameCompression } from '../../codec/useCurrentFrameCompression'
import { useAppSettings } from '../../../hooks/useAppSettings'
import { locateGlobeGridPoint } from '../data/globeCoordinateUtils'
import { useGlobeControls } from '../hooks/useGlobeControls'
import { useGlobeManifest } from '../hooks/useGlobeManifest'
import { useGlobeSelection } from '../hooks/useGlobeSelection'
import { useGlobeValues } from '../hooks/useGlobeValues'
import { useTccCloudLayer } from '../hooks/useTccCloudLayer'
import { useWeatherGlobeCatalog, useWeatherGlobeLayer } from '../hooks/useWeatherGlobe'
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
const earthTexturePath = 'data/globe/base/earth-blue-marble.jpg'

export default function AtmosphereGlobe({
  manifestUrl = defaultManifestUrl,
  defaultChannel = 't2m',
  defaultTimestamp,
  defaultGrid = '0p25',
  availableModes = ['original'],
  researchDefaults = {},
}: AtmosphereGlobeProps) {
  const cardRef = useRef<HTMLElement>(null)
  const { settings } = useAppSettings()
  const { manifest, loading: manifestLoading, error: manifestError, reload } =
    useGlobeManifest(manifestUrl)
  const weatherApiEnabled = settings.api.enabled
  const weatherCatalog = useWeatherGlobeCatalog(weatherApiEnabled)
  const activeModes: GlobeMode[] = weatherApiEnabled ? ['original'] : availableModes
  const demoEnabled = import.meta.env.VITE_ENABLE_GLOBE_DEMO === 'true'
  const manifestFrames = useMemo(
    () =>
      (manifest?.frames ?? []).filter(
        (frame) => frame.source !== 'demo' || demoEnabled,
      ),
    [demoEnabled, manifest?.frames],
  )
  const frames = weatherApiEnabled ? weatherCatalog.frames : manifestFrames
  const selection = useGlobeSelection(frames, activeModes, {
    channel: defaultChannel,
    timestamp: defaultTimestamp,
    grid: defaultGrid,
    displayMode: weatherApiEnabled ? 'data' : undefined,
    ...researchDefaults,
  })
  const { autoRotate, setAutoRotate, reducedMotion, resetSignal, resetView } =
    useGlobeControls()
  const [selectedCoordinates, setSelectedCoordinates] = useState<GlobeCoordinates | null>(null)
  const [textureLoading, setTextureLoading] = useState(false)
  const [textureError, setTextureError] = useState<string | null>(null)
  const [textureRevision, setTextureRevision] = useState(0)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const isEarthMode = selection.displayMode === 'earth'
  const frameCompression = useCurrentFrameCompression(
    selection.timestamp || undefined,
    weatherApiEnabled && !isEarthMode,
  )
  const weatherLayer = useWeatherGlobeLayer(selection.frame, weatherApiEnabled && !isEarthMode)
  const frame = weatherApiEnabled ? weatherLayer.layer?.frame ?? selection.frame : selection.frame
  const cloudWeatherLayer = useWeatherGlobeLayer(
    selection.cloudFrame,
    weatherApiEnabled && isEarthMode,
  )
  const cloudLayer = cloudWeatherLayer.layer
  const cloudValues =
    cloudLayer && cloudLayer.frame.id === selection.cloudFrame?.id
      ? cloudLayer.values
      : undefined
  const cloudState = useTccCloudLayer(
    selection.cloudFrame,
    isEarthMode,
    cloudValues,
    cloudWeatherLayer.loading,
    cloudWeatherLayer.error,
  )
  const valuesState = useGlobeValues(
    !weatherApiEnabled && !isEarthMode && selectedCoordinates ? frame?.valuesUrl : undefined,
    frame ? frame.width * frame.height : undefined,
  )
  const activeValues = weatherApiEnabled ? weatherLayer.layer?.values : valuesState.values
  const selectedPoint = useMemo<GlobeGridPoint | null>(
    () =>
      !isEarthMode && selectedCoordinates && frame
        ? locateGlobeGridPoint(selectedCoordinates, frame, activeValues ?? undefined)
        : null,
    [activeValues, frame, isEarthMode, selectedCoordinates],
  )
  const textureUrl = isEarthMode
    ? resolvePublicAssetUrl(earthTexturePath)
    : weatherApiEnabled
      ? weatherLayer.layer?.textureUrl
      : frame
      ? resolvePublicAssetUrl(frame.textureUrl)
      : undefined
  const firstAvailableFrame =
    frames.find((candidate) => candidate.mode === selection.mode) ?? frames[0]
  const statusFrame = isEarthMode
    ? (cloudState.loadedFrame ?? selection.cloudFrame)
    : frame

  useEffect(() => {
    setTextureError(null)
  }, [textureUrl])

  useEffect(() => {
    if (isEarthMode) setSelectedCoordinates(null)
  }, [isEarthMode])

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
      className={`ui-surface min-w-0 max-w-full overflow-hidden border font-sans ${
        isFullscreen
          ? 'fixed inset-0 z-[100] overflow-auto rounded-none border-0 bg-[var(--background)] p-4'
          : 'rounded-[20px]'
      }`}
    >
      <div className="flex flex-col gap-4 border-b border-[var(--border)] px-5 py-5 sm:flex-row sm:items-start sm:justify-between lg:px-6">
        <div className="flex min-w-0 items-start gap-3.5">
          <span className="ui-accent-icon flex h-10 w-10 shrink-0 items-center justify-center rounded-xl">
            <Globe2 className="h-[18px] w-[18px]" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2.5">
              <h2 className="text-[19px] font-bold leading-[1.25] tracking-[-0.025em] text-[var(--text-primary)] sm:text-[22px]">
                Глобальное состояние атмосферы ERA5
              </h2>
              {weatherApiEnabled && !isEarthMode ? (
                <span className="ui-accent-badge rounded-md px-2 py-1 text-[11px] font-semibold">
                  ERA5 Zarr · реальные данные
                </span>
              ) : statusFrame?.source === 'demo' ? (
                <span className="rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-[11px] font-semibold text-amber-700">
                  Демо-данные
                </span>
              ) : null}
            </div>
            <p className="mt-1 text-[14px] font-normal leading-5 text-[#667085]">
              {getLayerDescription(selection.displayMode, selection.mode)}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-2 sm:justify-start">
          {weatherApiEnabled && !isEarthMode ? (
            <FrameCompressionButton
              timestamp={selection.timestamp || undefined}
              frameReady={Boolean(frame) && !weatherLayer.loading && !weatherLayer.error}
              serviceLoading={frameCompression.serviceLoading}
              serviceReady={frameCompression.service?.ready === true}
              processing={frameCompression.processing}
              onCompress={frameCompression.compressCurrentFrame}
            />
          ) : null}
          <IconButton label="Сбросить вид" title="Вернуть вид на Европу и Африку" onClick={resetView}>
            <RotateCcw className="h-[18px] w-[18px]" aria-hidden="true" />
          </IconButton>
          <IconButton
            label={isFullscreen ? 'Выйти из полноэкранного режима' : 'Во весь экран'}
            title={isFullscreen ? 'Выйти из полноэкранного режима' : 'Во весь экран'}
            onClick={() => void toggleFullscreen()}
          >
            {isFullscreen ? (
              <Minimize2 className="h-[18px] w-[18px]" aria-hidden="true" />
            ) : (
              <Expand className="h-[18px] w-[18px]" aria-hidden="true" />
            )}
          </IconButton>
        </div>
      </div>

      <GlobeFilters
        selection={selection}
        availableModes={activeModes}
        autoRotate={autoRotate}
        onAutoRotateChange={setAutoRotate}
      />

      {weatherApiEnabled && !isEarthMode ? (
        <FrameCompressionNotice
          processing={frameCompression.processing}
          job={frameCompression.job}
          error={frameCompression.error}
          service={frameCompression.service}
          serviceError={frameCompression.serviceError}
          onRetryService={frameCompression.retryService}
        />
      ) : null}

      {selection.displayMode === 'data' && frame ? (
        <GlobeColorLegend frame={frame} mode={selection.mode} />
      ) : null}

      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-4 p-3 sm:p-5 lg:gap-6 lg:p-6 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="globe-stage relative min-h-[420px] min-w-0 overflow-hidden rounded-2xl border sm:min-h-[500px] lg:min-h-[520px] xl:min-h-[580px]">
          <GlobeCanvas
            displayMode={selection.displayMode}
            textureUrl={textureUrl}
            cloudTexture={cloudState.texture}
            textureRevision={textureRevision}
            autoRotate={autoRotate}
            reducedMotion={reducedMotion}
            isFullscreen={isFullscreen}
            resetSignal={resetSignal}
            onPointSelect={setSelectedCoordinates}
            onTextureLoadingChange={setTextureLoading}
            onTextureError={setTextureError}
          />

          {!isEarthMode && (weatherApiEnabled ? weatherCatalog.loading : manifestLoading) ? (
            <GlobeLoadingOverlay message="Загрузка каталога реальных слоёв ERA5…" />
          ) : null}
          {!isEarthMode && weatherLayer.loading ? (
            <GlobeLoadingOverlay message="Загрузка поля ERA5 из Zarr…" />
          ) : null}
          {textureLoading ? (
            <GlobeLoadingOverlay
              message={
                isEarthMode
                  ? 'Загрузка поверхности Земли…'
                  : 'Загрузка слоя атмосферы…'
              }
            />
          ) : null}
          {!isEarthMode && (weatherApiEnabled ? weatherCatalog.error : manifestError) ? (
            <GlobeErrorState
              message="Не удалось загрузить каталог атмосферы ERA5"
              details={weatherApiEnabled ? weatherCatalog.error ?? '' : manifestError ?? ''}
              onRetry={weatherApiEnabled ? weatherCatalog.reload : reload}
            />
          ) : null}
          {!isEarthMode && weatherLayer.error ? (
            <GlobeErrorState
              message="Не удалось загрузить поле ERA5"
              details={weatherLayer.error}
              onRetry={weatherLayer.reload}
            />
          ) : null}
          {textureError ? (
            <GlobeErrorState
              message={
                isEarthMode
                  ? 'Не удалось загрузить поверхность Земли'
                  : 'Не удалось загрузить слой атмосферы'
              }
              details={textureError}
              onRetry={retryTexture}
            />
          ) : null}
          {!isEarthMode &&
          !(weatherApiEnabled ? weatherCatalog.loading : manifestLoading) &&
          !(weatherApiEnabled ? weatherCatalog.error : manifestError) &&
          !textureError &&
          !frame ? (
            <GlobeEmptyState
              message={
                frames.length === 0
                  ? 'Слой ERA5 ещё не доступен'
                  : 'Для выбранной комбинации данных нет'
              }
              details={
                frames.length === 0
                  ? weatherApiEnabled
                    ? 'Проверьте, что API доступен на http://localhost:8000.'
                    : demoEnabled
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
          {isEarthMode && !textureError ? (
            <EarthCloudStatus
              manifestLoading={manifestLoading}
              manifestError={manifestError}
              hasCloudFrame={Boolean(selection.cloudFrame)}
              cloudLoading={cloudState.loading}
              cloudError={cloudState.error}
              onManifestRetry={reload}
              onCloudRetry={cloudState.retry}
            />
          ) : null}
        </div>

        <GlobeParameterPanel
          displayMode={selection.displayMode}
          frame={frame}
          cloudFrame={cloudState.loadedFrame}
          cloudLoading={cloudState.loading}
          cloudError={
            cloudState.error ??
            (!manifestLoading && !selection.cloudFrame
              ? 'В manifest нет исходного кадра общей облачности TCC.'
              : null)
          }
          onCloudRetry={cloudState.error ? cloudState.retry : undefined}
          point={selectedPoint}
          valuesLoading={weatherApiEnabled ? weatherLayer.loading : valuesState.loading}
          valuesError={weatherApiEnabled ? weatherLayer.error : valuesState.error}
        />
      </div>

      {frameCompression.job ? (
        <FrameCompressionReport
          job={frameCompression.job}
          jobTimestamp={frameCompression.jobTimestamp}
          codecBaseUrl={settings.services.codecBaseUrl}
        />
      ) : null}

      {reducedMotion ? (
        <p className="sr-only">
          Автовращение отключено в соответствии с настройкой уменьшения движения.
        </p>
      ) : null}
    </section>
  )
}

function getLayerDescription(
  displayMode: 'earth' | 'data',
  mode: GlobeMode,
) {
  if (displayMode === 'earth') {
    return 'Обычная Земля с облачностью TCC для выбранного момента времени'
  }
  if (mode === 'reconstruction') {
    return 'Восстановленное моделью поле в выбранный момент времени'
  }
  if (mode === 'absolute-error') {
    return 'Абсолютная разница между ERA5 и восстановлением модели'
  }
  return 'Исходные данные реанализа в выбранный момент времени'
}

function EarthCloudStatus({
  manifestLoading,
  manifestError,
  hasCloudFrame,
  cloudLoading,
  cloudError,
  onManifestRetry,
  onCloudRetry,
}: {
  manifestLoading: boolean
  manifestError: string | null
  hasCloudFrame: boolean
  cloudLoading: boolean
  cloudError: string | null
  onManifestRetry: () => void
  onCloudRetry: () => void
}) {
  if (manifestLoading || cloudLoading) {
    return (
      <div
        className="ui-accent-badge absolute left-3 top-3 z-20 rounded-xl px-3.5 py-2.5 text-[12px] font-medium shadow-[var(--shadow-soft)]"
        aria-live="polite"
      >
        {manifestLoading
          ? 'Поиск доступных кадров TCC…'
          : 'Загрузка облачности для выбранного времени…'}
      </div>
    )
  }

  const message = manifestError
    ? `Каталог TCC недоступен: ${manifestError}`
    : cloudError
      ? `Облачность временно недоступна: ${cloudError}`
      : !hasCloudFrame
        ? 'В manifest пока нет исходного кадра общей облачности TCC.'
        : null
  if (!message) return null

  return (
    <div
      className="absolute left-3 top-3 z-20 max-w-[calc(100%-1.5rem)] rounded-xl border border-amber-200 bg-[var(--surface)] px-3.5 py-2.5 text-[12px] leading-5 text-amber-950 shadow-[var(--shadow-soft)]"
      role="status"
      aria-live="polite"
    >
      <span>{message}</span>
      {(manifestError || cloudError) ? (
        <button
          type="button"
          onClick={manifestError ? onManifestRetry : onCloudRetry}
          className="ml-2 font-semibold underline decoration-amber-400 underline-offset-2"
        >
          Повторить
        </button>
      ) : null}
    </div>
  )
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
      className="ui-button-ghost ui-focus-ring inline-flex h-10 w-10 items-center justify-center rounded-xl"
    >
      {children}
    </button>
  )
}
