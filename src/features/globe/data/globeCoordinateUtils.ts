import type { GlobeCoordinates, GlobeFrameAsset, GlobeGridPoint } from '../types/globe'

const FULL_CIRCLE = 360

export function normalizeLongitude(longitude: number) {
  return ((longitude + 180) % FULL_CIRCLE + FULL_CIRCLE) % FULL_CIRCLE - 180
}

export function locateGlobeGridPoint(
  coordinates: GlobeCoordinates,
  frame: GlobeFrameAsset,
  values?: Float32Array,
): GlobeGridPoint {
  const latitude = Math.max(-90, Math.min(90, coordinates.latitude))
  const longitude = normalizeLongitude(coordinates.longitude)
  const latitudePosition =
    frame.latitudeOrder === 'south-to-north' ? (latitude + 90) / 180 : (90 - latitude) / 180
  const row = Math.max(0, Math.min(frame.height - 1, Math.round(latitudePosition * (frame.height - 1))))

  const longitudeStart = frame.longitudeRange === '0-360' ? 0 : -180
  const normalizedForFrame =
    frame.longitudeRange === '0-360'
      ? ((longitude % FULL_CIRCLE) + FULL_CIRCLE) % FULL_CIRCLE
      : longitude - longitudeStart
  const column = Math.round((normalizedForFrame / FULL_CIRCLE) * frame.width) % frame.width
  const flatIndex = row * frame.width + column
  const candidate = values?.[flatIndex]
  const isNoData =
    candidate === undefined ||
    !Number.isFinite(candidate) ||
    (frame.noDataValue !== undefined && Object.is(candidate, frame.noDataValue))

  return {
    latitude,
    longitude,
    row,
    column,
    flatIndex,
    value: isNoData ? undefined : candidate,
  }
}
