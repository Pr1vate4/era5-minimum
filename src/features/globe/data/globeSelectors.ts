import { globeParameterConfigs } from './globeParameterConfig'
import type {
  GlobeChannel,
  GlobeFrameAsset,
  GlobeGrid,
  GlobeManifest,
  GlobeMode,
  GlobeSource,
  PressureLevel,
} from '../types/globe'

const modes = new Set<GlobeMode>(['original', 'reconstruction', 'absolute-error'])
const grids = new Set<GlobeGrid>(['0p25', '0p5'])
const channels = new Set<GlobeChannel>(globeParameterConfigs.map(({ channel }) => channel))
const sources = new Set<GlobeSource>(['ERA5', 'model', 'demo'])
const pressureLevels = new Set<PressureLevel>([1000, 925, 850, 700])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function optionalFiniteNumber(value: unknown) {
  return value === undefined || (typeof value === 'number' && Number.isFinite(value))
}

function optionalString(value: unknown) {
  return value === undefined || typeof value === 'string'
}

function isFrameAsset(value: unknown): value is GlobeFrameAsset {
  if (!isRecord(value)) return false
  if (typeof value.id !== 'string' || !value.id) return false
  if (!modes.has(value.mode as GlobeMode)) return false
  if (!channels.has(value.channel as GlobeChannel)) return false
  if (!grids.has(value.grid as GlobeGrid)) return false
  if (!sources.has(value.source as GlobeSource)) return false
  if (typeof value.timestamp !== 'string' || Number.isNaN(Date.parse(value.timestamp))) return false
  if (typeof value.textureUrl !== 'string' || !value.textureUrl) return false
  if (!Number.isInteger(value.width) || Number(value.width) <= 0) return false
  if (!Number.isInteger(value.height) || Number(value.height) <= 0) return false
  if (typeof value.unit !== 'string') return false
  if (!optionalString(value.valuesUrl) || !optionalString(value.runId) || !optionalString(value.checkpoint)) return false
  if (
    !optionalFiniteNumber(value.min) ||
    !optionalFiniteNumber(value.max) ||
    !optionalFiniteNumber(value.mean) ||
    !optionalFiniteNumber(value.compressionRatio) ||
    !optionalFiniteNumber(value.trainFrames) ||
    !optionalFiniteNumber(value.noDataValue)
  ) {
    return false
  }
  if (value.level !== undefined && !pressureLevels.has(value.level as PressureLevel)) return false
  if (
    value.latitudeOrder !== undefined &&
    value.latitudeOrder !== 'north-to-south' &&
    value.latitudeOrder !== 'south-to-north'
  ) {
    return false
  }
  if (
    value.longitudeRange !== undefined &&
    value.longitudeRange !== '0-360' &&
    value.longitudeRange !== '-180-180'
  ) {
    return false
  }
  if (value.valueEncoding !== undefined && value.valueEncoding !== 'float32-le') return false
  if (value.normalization !== undefined) {
    if (!isRecord(value.normalization)) return false
    if (
      typeof value.normalization.mean !== 'number' ||
      !Number.isFinite(value.normalization.mean) ||
      typeof value.normalization.std !== 'number' ||
      !Number.isFinite(value.normalization.std) ||
      value.normalization.std <= 0
    ) {
      return false
    }
  }
  return true
}

export function parseGlobeManifest(value: unknown) {
  if (!isRecord(value) || !Array.isArray(value.frames)) {
    throw new Error('Manifest глобуса должен содержать массив frames.')
  }

  const frames = value.frames.filter(isFrameAsset)
  return {
    manifest: {
      version: typeof value.version === 'number' ? value.version : 1,
      generatedAt: typeof value.generatedAt === 'string' ? value.generatedAt : undefined,
      frames,
    } satisfies GlobeManifest,
    invalidFrameCount: value.frames.length - frames.length,
  }
}

export function uniqueValues<T extends string | number>(
  values: Array<T | undefined>,
) {
  return Array.from(new Set(values.filter((value): value is T => value !== undefined)))
}
