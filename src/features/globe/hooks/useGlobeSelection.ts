import { useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { globeParameterConfigs, isPressureChannel } from '../data/globeParameterConfig'
import { uniqueValues } from '../data/globeSelectors'
import {
  NO_PARAMETER_VALUE,
  type GlobeChannel,
  type GlobeDisplayMode,
  type GlobeFrameAsset,
  type GlobeGrid,
  type GlobeMode,
  type GlobeResearchDefaults,
  type PressureLevel,
} from '../types/globe'

type SelectionDefaults = GlobeResearchDefaults & {
  channel?: string
  timestamp?: string
  grid?: string
  displayMode?: GlobeDisplayMode
}

const gridOrder: GlobeGrid[] = ['0p25', '0p5']
const pressureLevelOrder: PressureLevel[] = [1000, 925, 850, 700]
const scientificUrlKeys = [
  'channel',
  'level',
  'mode',
  'runId',
  'trainFrames',
  'compressionRatio',
  'checkpoint',
] as const

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
  const searchParamsKey = searchParams.toString()
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
      : firstMatching(requestedRunId ?? defaults.runId ?? null, runIds, String)

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
  const urlChannel = searchParams.get('channel')
  const requestedChannel = urlChannel ?? defaults.channel
  const channel =
    enabledChannels.find((candidate) => candidate === requestedChannel) ??
    enabledChannels[0] ??
    't2m'
  const hasValidUrlChannel = enabledChannels.some((candidate) => candidate === urlChannel)
  const requestedView = searchParams.get('view')
  const displayMode: GlobeDisplayMode =
    requestedView !== 'earth' &&
    (hasValidUrlChannel || (requestedView === null && defaults.displayMode === 'data' && enabledChannels.length > 0))
      ? 'data'
      : 'earth'

  const channelFrames = researchFrames.filter((frame) => frame.channel === channel)
  const availableGrids = new Set(channelFrames.map((frame) => frame.grid))
  const dataGridOptions = gridOrder.map((value) => ({
    value,
    available: availableGrids.has(value),
  }))
  const enabledGrids = dataGridOptions
    .filter((option) => option.available)
    .map((option) => option.value)
  const requestedGrid = normalizeGrid(searchParams.get('grid') ?? defaults.grid)
  const dataGrid = enabledGrids.includes(requestedGrid as GlobeGrid)
    ? (requestedGrid as GlobeGrid)
    : (enabledGrids[0] ?? '0p25')
  const gridFrames = channelFrames.filter((frame) => frame.grid === dataGrid)

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

  const dataTimestampOptions = uniqueValues(levelFrames.map((frame) => frame.timestamp)).sort()
  const requestedTimestamp = searchParams.get('timestamp') ?? defaults.timestamp
  const dataTimestamp =
    dataTimestampOptions.find((candidate) => candidate === requestedTimestamp) ??
    dataTimestampOptions[0] ??
    ''
  const dataFrame = levelFrames.find((candidate) => candidate.timestamp === dataTimestamp)

  const allOriginalTccFrames = useMemo(
    () => frames.filter((frame) => frame.mode === 'original' && frame.channel === 'tcc'),
    [frames],
  )
  const era5TccFrames = allOriginalTccFrames.filter((frame) => frame.source === 'ERA5')
  const tccFrames = era5TccFrames.length > 0 ? era5TccFrames : allOriginalTccFrames
  const exactTimestampFrames = requestedTimestamp
    ? tccFrames.filter((frame) => frame.timestamp === requestedTimestamp)
    : []
  const earthGridCandidates =
    exactTimestampFrames.length > 0 ? exactTimestampFrames : tccFrames
  const earthAvailableGrids = new Set(earthGridCandidates.map((frame) => frame.grid))
  const earthGrid =
    gridOrder.find(
      (candidate) =>
        candidate === requestedGrid && earthAvailableGrids.has(candidate),
    ) ??
    gridOrder.find((candidate) => earthAvailableGrids.has(candidate)) ??
    earthGridCandidates[0]?.grid ??
    '0p25'
  const earthGridFrames = tccFrames.filter((frame) => frame.grid === earthGrid)
  const earthTimestampOptions = uniqueValues(
    earthGridFrames.map((frame) => frame.timestamp),
  ).sort()
  const earthTimestamp = findNearestTimestamp(
    requestedTimestamp,
    earthTimestampOptions,
  )
  const cloudFrame = earthGridFrames.find(
    (candidate) => candidate.timestamp === earthTimestamp,
  )
  const framesAtEarthTimestamp = tccFrames.filter(
    (frame) => frame.timestamp === earthTimestamp,
  )
  const gridsAtEarthTimestamp = new Set(
    framesAtEarthTimestamp.map((frame) => frame.grid),
  )
  const earthGridOptions = gridOrder.map((value) => ({
    value,
    available: gridsAtEarthTimestamp.has(value),
  }))

  useEffect(() => {
    const next = new URLSearchParams(searchParamsKey)
    if (displayMode === 'earth') {
      next.set('view', 'earth')
      scientificUrlKeys.forEach((key) => next.delete(key))
      if (cloudFrame) {
        next.set('timestamp', cloudFrame.timestamp)
        next.set('grid', cloudFrame.grid)
      }
    } else {
      next.set('view', 'data')
    }

    if (next.toString() !== searchParamsKey) {
      setSearchParams(next, { replace: true })
    }
  }, [cloudFrame, displayMode, searchParamsKey, setSearchParams])

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

  const setDisplayMode = (value: GlobeDisplayMode) => {
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        next.set('view', value)
        if (value === 'earth') {
          scientificUrlKeys.forEach((key) => next.delete(key))
          if (cloudFrame) {
            next.set('timestamp', cloudFrame.timestamp)
            next.set('grid', cloudFrame.grid)
          }
        } else {
          next.set('channel', channel)
          next.set('mode', mode)
        }
        return next
      },
      { replace: true },
    )
  }

  const setChannel = (value: GlobeChannel) => {
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        next.set('view', 'data')
        next.set('channel', value)
        return next
      },
      { replace: true },
    )
  }

  const selectFrame = (candidate: GlobeFrameAsset) => {
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        next.set('view', 'data')
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
    displayMode,
    mode,
    channel: channel as GlobeChannel,
    grid: displayMode === 'earth' ? earthGrid : dataGrid,
    level: displayMode === 'earth' ? undefined : level,
    timestamp: displayMode === 'earth' ? earthTimestamp : dataTimestamp,
    frame: displayMode === 'data' ? dataFrame : undefined,
    cloudFrame,
    channelOptions,
    gridOptions: displayMode === 'earth' ? earthGridOptions : dataGridOptions,
    levelOptions,
    timestampOptions:
      displayMode === 'earth' ? earthTimestampOptions : dataTimestampOptions,
    researchOptions: {
      runIds,
      trainFrames: trainFrameOptions,
      compressionRatios: compressionOptions,
      checkpoints: checkpointOptions,
    },
    research: { runId, trainFrames, compressionRatio, checkpoint },
    setDisplayMode,
    setMode: (value: GlobeMode) =>
      setSearchParams(
        (current) => {
          const next = new URLSearchParams(current)
          next.set('view', 'data')
          next.set('channel', channel)
          next.set('mode', value)
          return next
        },
        { replace: true },
      ),
    setChannel,
    setGrid: (value: GlobeGrid) => updateParam('grid', value),
    setLevel: (value: PressureLevel) => updateParam('level', value),
    setTimestamp: (value: string) => updateParam('timestamp', value),
    setRunId: (value: string) => updateParam('runId', value),
    setTrainFrames: (value: number) => updateParam('trainFrames', value),
    setCompressionRatio: (value: number) => updateParam('compressionRatio', value),
    setCheckpoint: (value: string) => updateParam('checkpoint', value),
    selectFrame,
    noParameterValue: NO_PARAMETER_VALUE,
  }
}

function normalizeGrid(value: string | undefined | null) {
  if (!value) return undefined
  if (value === '0.25deg' || value === '0.25°') return '0p25'
  if (value === '0.5deg' || value === '0.5°') return '0p5'
  return value
}

function findNearestTimestamp(
  requested: string | undefined,
  options: string[],
) {
  if (options.length === 0) return ''
  if (requested && options.includes(requested)) return requested
  if (!requested || Number.isNaN(Date.parse(requested))) return options[0]

  const requestedTime = Date.parse(requested)
  return options.reduce((nearest, candidate) => {
    const nearestDistance = Math.abs(Date.parse(nearest) - requestedTime)
    const candidateDistance = Math.abs(Date.parse(candidate) - requestedTime)
    return candidateDistance < nearestDistance ? candidate : nearest
  }, options[0])
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
