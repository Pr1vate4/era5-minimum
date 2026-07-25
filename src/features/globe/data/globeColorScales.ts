import { getGlobeParameterConfig, toDisplayValue } from './globeParameterConfig'
import type { GlobeFrameAsset, GlobeMode } from '../types/globe'

const errorColors = ['#FFF7ED', '#FED7AA', '#FB923C', '#EF4444', '#991B1B']

export function getGlobeColorScale(frame: GlobeFrameAsset | undefined, mode: GlobeMode) {
  if (!frame) {
    const fallback = getGlobeParameterConfig('t2m')
    return {
      colors: fallback.colors,
      min: fallback.fallbackRange[0],
      max: fallback.fallbackRange[1],
      unit: fallback.displayUnit,
    }
  }

  const configuration = getGlobeParameterConfig(frame.channel)
  const fallbackMin = mode === 'absolute-error' ? 0 : configuration.fallbackRange[0]
  const fallbackMax =
    mode === 'absolute-error'
      ? Math.max(Math.abs(configuration.fallbackRange[1] - configuration.fallbackRange[0]) * 0.15, 1)
      : configuration.fallbackRange[1]
  const convertedMin =
    frame.min === undefined ? fallbackMin : toDisplayValue(frame.channel, frame.min, frame.unit)
  const convertedMax =
    frame.max === undefined ? fallbackMax : toDisplayValue(frame.channel, frame.max, frame.unit)

  return {
    colors: mode === 'absolute-error' ? errorColors : configuration.colors,
    min: Number.isFinite(convertedMin) ? Math.max(mode === 'absolute-error' ? 0 : -Infinity, convertedMin) : fallbackMin,
    max: Number.isFinite(convertedMax) ? convertedMax : fallbackMax,
    unit: configuration.displayUnit,
  }
}

export function buildLegendTicks(min: number, max: number, count = 7) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [min]
  return Array.from({ length: count }, (_, index) => min + ((max - min) * index) / (count - 1))
}

export function formatLegendValue(value: number) {
  const absolute = Math.abs(value)
  if (absolute !== 0 && absolute < 0.01) return value.toExponential(1)
  if (absolute >= 1000) return value.toLocaleString('ru-RU', { maximumFractionDigits: 0 })
  if (absolute >= 10) return value.toLocaleString('ru-RU', { maximumFractionDigits: 1 })
  return value.toLocaleString('ru-RU', { maximumFractionDigits: 2 })
}
