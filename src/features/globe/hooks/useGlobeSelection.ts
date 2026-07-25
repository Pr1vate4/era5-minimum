import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { globeParameterConfigs, isPressureChannel } from '../data/globeParameterConfig'
import { uniqueValues } from '../data/globeSelectors'
import type {
  GlobeChannel,
  GlobeFrameAsset,
  GlobeGrid,
  GlobeMode,
  GlobeResearchDefaults,
  PressureLevel,
} from '../types/globe'

type SelectionDefaults = GlobeResearchDefaults & {
  channel?: string
  timestamp?: string
  grid?: string
}

const gridOrder: GlobeGrid[] = ['0p25', '0p5']
const pressureLevelOrder: PressureLevel[] = [1000, 925, 850, 700]

function firstMatching<T>(requested: string | null, options: T[], stringify: (value: T) => string) {
  return options.find((option) => stringify(option) === requested) ?? options[0]
}

function parseOptionalNumber(value: string | null) {
  if (value === null || value.trim() === '') return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

export function useGlobeSelection(
  frames: GlobeFrameAsset[],
  availableModes: GlobeMode[],
  defaults: SelectionDefaults,
) {
  const [searchParams, setSearchParams] = useSearchParams()
  const requestedMode = searchParams.get('mode') as GlobeMode | null
  const mode =
    requestedMode && availableModes.includes(requestedMode)
      ? requestedMode
      : (availableModes[0] ?? 'original')

  const modeFrames = useMemo(
    () => frames.filter((frame) => frame.mode === mode),
    [frames, mode],
  )

  const runIds = uniqueValues(modeFrames.map((frame) => frame.runId))
  const requestedRunId = searchParams.get('runId')
  const runId =
    mode === 'original'
      ? undefined
      : firstMatching(
          requestedRunId ?? defaults.runId ?? null,
          runIds,
          String,
        )

  const runFrames = modeFrames.filter(
    (frame) => mode === 'original' || !runId || frame.runId === runId,
  )
  const trainFrameOptions = uniqueValues(runFrames.map((frame) => frame.trainFrames)).sort(
    (left, right) => left - right,
  )
  const trainFrames =
    mode === 'original'
      ? undefined
      : firstMatching(
          searchParams.get('trainFrames') ?? String(defaults.trainFrames ?? ''),
          trainFrameOptions,
          String,
        )
  const trainFiltered = runFrames.filter(
    (frame) => mode === 'original' || trainFrames === undefined || frame.trainFrames === trainFrames,
  )
  const compressionOptions = uniqueValues(
    trainFiltered.map((frame) => frame.compressionRatio),
  ).sort((left, right) => left - right)
  const compressionRatio =
    mode === 'original'
      ? undefined
      : firstMatching(
          searchParams.get('compressionRatio') ?? String(defaults.compressionRatio ?? ''),
          compressionOptions,
          String,
        )
  const compressionFiltered = trainFiltered.filter(
    (frame) =>
      mode === 'original' ||
      compressionRatio === undefined ||
      frame.compressionRatio === compressionRatio,
  )
  const checkpointOptions = uniqueValues(compressionFiltered.map((frame) => frame.checkpoint))
  const checkpoint =
    mode === 'original'
      ? undefined
      : firstMatching(
          searchParams.get('checkpoint') ?? defaults.checkpoint ?? null,
          checkpointOptions,
          String,
        )
  const researchFrames = compressionFiltered.filter(
    (frame) => mode === 'original' || !checkpoint || frame.checkpoint === checkpoint,
  )

  const availableChannels = new Set(researchFrames.map((frame) => frame.channel))
  const channelOptions = globeParameterConfigs.map((configuration) => ({
    ...configuration,
    available: availableChannels.has(configuration.channel),
  }))
  const enabledChannels = channelOptions
    .filter((option) => option.available)
    .map((option) => option.channel)
  const requestedChannel = searchParams.get('channel') ?? defaults.channel
  const channel =
    enabledChannels.find((candidate) => candidate === requestedChannel) ??
    enabledChannels[0] ??
    't2m'
  const channelFrames = researchFrames.filter((frame) => frame.channel === channel)

  const availableGrids = new Set(channelFrames.map((frame) => frame.grid))
  const gridOptions = gridOrder.map((value) => ({ value, available: availableGrids.has(value) }))
  const enabledGrids = gridOptions.filter((option) => option.available).map((option) => option.value)
  const requestedGrid = normalizeGrid(searchParams.get('grid') ?? defaults.grid)
  const grid = enabledGrids.includes(requestedGrid as GlobeGrid)
    ? (requestedGrid as GlobeGrid)
    : (enabledGrids[0] ?? '0p25')
  const gridFrames = channelFrames.filter((frame) => frame.grid === grid)

  const levelOptions = isPressureChannel(channel)
    ? pressureLevelOrder.filter((level) => gridFrames.some((frame) => frame.level === level))
    : []
  const requestedLevel = parseOptionalNumber(searchParams.get('level'))
  const level = isPressureChannel(channel)
    ? ((requestedLevel && levelOptions.includes(requestedLevel as PressureLevel)
        ? requestedLevel
        : levelOptions.includes(850)
          ? 850
          : levelOptions[0]) as PressureLevel | undefined)
    : undefined
  const levelFrames = gridFrames.filter(
    (frame) => !isPressureChannel(channel) || frame.level === level,
  )

  const timestampOptions = uniqueValues(levelFrames.map((frame) => frame.timestamp)).sort()
  const requestedTimestamp = searchParams.get('timestamp') ?? defaults.timestamp
  const timestamp =
    timestampOptions.find((candidate) => candidate === requestedTimestamp) ??
    timestampOptions[0] ??
    ''
  const frame = levelFrames.find((candidate) => candidate.timestamp === timestamp)

  const updateParam = (key: string, value: string | number | undefined) => {
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        if (value === undefined || value === '') next.delete(key)
        else next.set(key, String(value))
        return next
      },
      { replace: true },
    )
  }

  const selectFrame = (candidate: GlobeFrameAsset) => {
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        next.set('mode', candidate.mode)
        next.set('channel', candidate.channel)
        next.set('timestamp', candidate.timestamp)
        next.set('grid', candidate.grid)
        setOptionalParam(next, 'level', candidate.level)
        setOptionalParam(next, 'runId', candidate.runId)
        setOptionalParam(next, 'trainFrames', candidate.trainFrames)
        setOptionalParam(next, 'compressionRatio', candidate.compressionRatio)
        setOptionalParam(next, 'checkpoint', candidate.checkpoint)
        return next
      },
      { replace: true },
    )
  }

  return {
    mode,
    channel: channel as GlobeChannel,
    grid,
    level,
    timestamp,
    frame,
    channelOptions,
    gridOptions,
    levelOptions,
    timestampOptions,
    researchOptions: {
      runIds,
      trainFrames: trainFrameOptions,
      compressionRatios: compressionOptions,
      checkpoints: checkpointOptions,
    },
    research: { runId, trainFrames, compressionRatio, checkpoint },
    setMode: (value: GlobeMode) => updateParam('mode', value),
    setChannel: (value: GlobeChannel) => updateParam('channel', value),
    setGrid: (value: GlobeGrid) => updateParam('grid', value),
    setLevel: (value: PressureLevel) => updateParam('level', value),
    setTimestamp: (value: string) => updateParam('timestamp', value),
    setRunId: (value: string) => updateParam('runId', value),
    setTrainFrames: (value: number) => updateParam('trainFrames', value),
    setCompressionRatio: (value: number) => updateParam('compressionRatio', value),
    setCheckpoint: (value: string) => updateParam('checkpoint', value),
    selectFrame,
  }
}

function normalizeGrid(value: string | undefined | null) {
  if (!value) return undefined
  if (value === '0.25deg' || value === '0.25°') return '0p25'
  if (value === '0.5deg' || value === '0.5°') return '0p5'
  return value
}

function setOptionalParam(
  params: URLSearchParams,
  key: string,
  value: string | number | undefined,
) {
  if (value === undefined) params.delete(key)
  else params.set(key, String(value))
}

export type GlobeSelection = ReturnType<typeof useGlobeSelection>
