import { useEffect, useState } from 'react'
import { buildApiUrl } from '../../../app/settings'
import { useAppSettings } from '../../../hooks/useAppSettings'
import { getGlobeParameterConfig } from '../data/globeParameterConfig'
import type { GlobeChannel, GlobeFrameAsset, PressureLevel } from '../types/globe'

const DISPLAY_WIDTH = 720
const DISPLAY_HEIGHT = 360

type WeatherVariable = {
  logical_name: string
  unit: string
  kind: 'surface' | 'pressure'
  default_level: number | null
}

type WeatherLayerResponse = {
  variable: string
  timestamp: string
  level: number | null
  unit: string
  values: Array<Array<number | null>>
  mask: boolean[][]
  minimum: number
  maximum: number
  shape: [number, number]
  is_mock: boolean
  visualization_sampling: string
}

export type LiveWeatherLayer = {
  frame: GlobeFrameAsset
  values: Float32Array
  textureUrl: string
  samplingNote: string
}

function toFrame(variable: WeatherVariable, timestamp: string): GlobeFrameAsset | null {
  const pressure = variable.logical_name.match(/^([TUVZQ])(1000|925|850|700)$/)
  const channel = (pressure ? pressure[1] : variable.logical_name) as GlobeChannel
  const level = pressure ? Number(pressure[2]) as PressureLevel : undefined
  if (!getGlobeParameterConfig(channel)) return null

  return {
    id: `weather:${variable.logical_name}:${timestamp}`,
    mode: 'original',
    channel,
    level,
    timestamp,
    grid: '0p5',
    textureUrl: '',
    width: DISPLAY_WIDTH,
    height: DISPLAY_HEIGHT,
    unit: variable.unit,
    source: 'ERA5',
    latitudeOrder: 'north-to-south',
    longitudeRange: '0-360',
    apiVariable: variable.logical_name,
  }
}

export function useWeatherGlobeCatalog(enabled: boolean) {
  const { settings } = useAppSettings()
  const { baseUrl, timeoutMs } = settings.api
  const [frames, setFrames] = useState<GlobeFrameAsset[]>([])
  const [loading, setLoading] = useState(enabled)
  const [error, setError] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)

  useEffect(() => {
    if (!enabled) {
      setFrames([])
      setLoading(false)
      setError(null)
      return
    }

    const controller = new AbortController()
    let active = true
    let timedOut = false
    const timeoutId = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, timeoutMs)

    setLoading(true)
    setError(null)
    Promise.all([
      fetch(buildApiUrl(baseUrl, '/api/v1/variables'), { signal: controller.signal }),
      fetch(buildApiUrl(baseUrl, '/api/v1/timestamps'), { signal: controller.signal }),
    ])
      .then(async ([variablesResponse, timestampsResponse]) => {
        if (!variablesResponse.ok || !timestampsResponse.ok) {
          throw new Error(`HTTP ${variablesResponse.ok ? timestampsResponse.status : variablesResponse.status}`)
        }
        return Promise.all([
          variablesResponse.json() as Promise<WeatherVariable[]>,
          timestampsResponse.json() as Promise<string[]>,
        ])
      })
      .then(([variables, timestamps]) => {
        if (!active) return
        const nextFrames = variables.flatMap((variable) =>
          timestamps.flatMap((timestamp) => {
            const frame = toFrame(variable, timestamp)
            return frame ? [frame] : []
          }),
        )
        setFrames(nextFrames)
      })
      .catch((caught: unknown) => {
        if (!active) return
        setFrames([])
        setError(
          timedOut
            ? `API не ответил за ${timeoutMs / 1000} с.`
            : caught instanceof Error
              ? caught.message
              : 'Не удалось загрузить каталог ERA5.',
        )
      })
      .finally(() => {
        window.clearTimeout(timeoutId)
        if (active) setLoading(false)
      })

    return () => {
      active = false
      window.clearTimeout(timeoutId)
      controller.abort()
    }
  }, [baseUrl, enabled, revision, timeoutMs])

  return { frames, loading, error, reload: () => setRevision((value) => value + 1) }
}

export function useWeatherGlobeLayer(frame: GlobeFrameAsset | undefined, enabled: boolean) {
  const { settings } = useAppSettings()
  const { baseUrl, timeoutMs } = settings.api
  const [layer, setLayer] = useState<LiveWeatherLayer | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)

  useEffect(() => {
    if (!enabled || !frame?.apiVariable) {
      setLayer(null)
      setLoading(false)
      setError(null)
      return
    }
    const controller = new AbortController()
    let active = true
    let timedOut = false
    const timeoutId = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, timeoutMs)
    const params = new URLSearchParams({
      variable: frame.apiVariable,
      timestamp: frame.timestamp,
      target_width: String(DISPLAY_WIDTH),
      target_height: String(DISPLAY_HEIGHT),
    })
    if (frame.level) params.set('level', String(frame.level))
    setLoading(true)
    setError(null)
    fetch(buildApiUrl(baseUrl, `/api/v1/layers?${params}`), {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`)
        return response.json() as Promise<WeatherLayerResponse>
      })
      .then((response) => {
        if (!active) return
        if (response.is_mock) throw new Error('API вернул mock-слой вместо реальных данных.')
        const values = new Float32Array(response.shape[0] * response.shape[1])
        let sum = 0
        let count = 0
        response.values.forEach((row, rowIndex) => row.forEach((value, columnIndex) => {
          const index = rowIndex * response.shape[1] + columnIndex
          if (!response.mask[rowIndex]?.[columnIndex] || value === null || !Number.isFinite(value)) {
            values[index] = Number.NaN
            return
          }
          values[index] = value
          sum += value
          count += 1
        }))
        const liveFrame: GlobeFrameAsset = {
          ...frame,
          width: response.shape[1],
          height: response.shape[0],
          unit: response.unit,
          min: response.minimum,
          max: response.maximum,
          mean: count > 0 ? sum / count : undefined,
        }
        setLayer({
          frame: liveFrame,
          values,
          textureUrl: renderLayerTexture(liveFrame, values),
          samplingNote: response.visualization_sampling,
        })
      })
      .catch((caught: unknown) => {
        if (!active) return
        setLayer(null)
        setError(
          timedOut
            ? `API не ответил за ${timeoutMs / 1000} с.`
            : caught instanceof Error
              ? caught.message
              : 'Не удалось загрузить слой ERA5.',
        )
      })
      .finally(() => {
        window.clearTimeout(timeoutId)
        if (active) setLoading(false)
      })

    return () => {
      active = false
      window.clearTimeout(timeoutId)
      controller.abort()
    }
  }, [baseUrl, enabled, frame?.apiVariable, frame?.level, frame?.timestamp, revision, timeoutMs])

  return { layer, loading, error, reload: () => setRevision((value) => value + 1) }
}

function renderLayerTexture(frame: GlobeFrameAsset, values: Float32Array) {
  const canvas = document.createElement('canvas')
  canvas.width = frame.width
  canvas.height = frame.height
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Браузер не поддерживает Canvas 2D.')
  const image = context.createImageData(frame.width, frame.height)
  const configuration = getGlobeParameterConfig(frame.channel)
  const min = frame.min ?? configuration.fallbackRange[0]
  const max = frame.max ?? configuration.fallbackRange[1]
  // Surface textures use -180..180 ordering. Rotate ERA5's 0..360 columns so
  // masked coastlines (most visibly SST) align with the GeoJSON coastline.
  const longitudeShift =
    frame.longitudeRange === '0-360' ? Math.floor(frame.width / 2) : 0
  values.forEach((value, sourceIndex) => {
    const sourceRow = Math.floor(sourceIndex / frame.width)
    const sourceColumn = sourceIndex % frame.width
    const targetColumn =
      (sourceColumn - longitudeShift + frame.width) % frame.width
    const offset = (sourceRow * frame.width + targetColumn) * 4
    if (!Number.isFinite(value)) {
      image.data[offset + 3] = 0
      return
    }
    const [red, green, blue] = colorAt(configuration.colors, (value - min) / Math.max(max - min, Number.EPSILON))
    image.data[offset] = red
    image.data[offset + 1] = green
    image.data[offset + 2] = blue
    image.data[offset + 3] = 255
  })
  context.putImageData(image, 0, 0)
  return canvas.toDataURL('image/png')
}

function colorAt(colors: readonly string[], position: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, position))
  const scaled = clamped * (colors.length - 1)
  const left = Math.floor(scaled)
  const right = Math.min(colors.length - 1, left + 1)
  const amount = scaled - left
  const start = hexToRgb(colors[left])
  const end = hexToRgb(colors[right])
  return [0, 1, 2].map((index) => Math.round(start[index] + (end[index] - start[index]) * amount)) as [number, number, number]
}

function hexToRgb(color: string): [number, number, number] {
  const value = Number.parseInt(color.slice(1), 16)
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255]
}
