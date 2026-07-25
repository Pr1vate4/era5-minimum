import * as THREE from 'three'
import type { GlobeFrameAsset } from '../types/globe'

export const CLOUD_MIN_VISIBLE_COVER = 0.03
export const CLOUD_ALPHA_GAMMA = 1.15
// TCC describes coverage, not an opaque satellite photograph. Keep continents
// legible even where coverage reaches one.
export const CLOUD_MAX_OPACITY = 0.45

type CreateTccCloudTextureOptions = {
  frame: GlobeFrameAsset
  values: Float32Array
}

export function createTccCloudTexture({
  frame,
  values,
}: CreateTccCloudTextureOptions) {
  const expectedLength = frame.width * frame.height
  if (values.length !== expectedLength) {
    throw new Error(
      `TCC содержит ${values.length} значений, ожидалось ${expectedLength}.`,
    )
  }

  const pixels = new Uint8Array(expectedLength * 4)
  const usesPercentScale = isPercentScale(frame)
  const longitudeShift =
    frame.longitudeRange === '0-360' ? Math.floor(frame.width / 2) : 0

  for (let sourceRow = 0; sourceRow < frame.height; sourceRow += 1) {
    const targetRow =
      frame.latitudeOrder === 'south-to-north'
        ? sourceRow
        : frame.height - sourceRow - 1

    for (let sourceColumn = 0; sourceColumn < frame.width; sourceColumn += 1) {
      const sourceIndex = sourceRow * frame.width + sourceColumn
      const targetColumn =
        (sourceColumn - longitudeShift + frame.width) % frame.width
      const targetIndex = targetRow * frame.width + targetColumn
      const pixelOffset = targetIndex * 4
      const value = values[sourceIndex]

      pixels[pixelOffset] = 255
      pixels[pixelOffset + 1] = 255
      pixels[pixelOffset + 2] = 255
      pixels[pixelOffset + 3] = isMissingValue(value, frame)
        ? 0
        : Math.round(calculateCloudAlpha(normalizeTcc(value, usesPercentScale)) * 255)
    }
  }

  const texture = new THREE.DataTexture(
    pixels,
    frame.width,
    frame.height,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  )
  texture.name = `TCC clouds · ${frame.id}`
  texture.colorSpace = THREE.NoColorSpace
  texture.flipY = false
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.ClampToEdgeWrapping
  texture.repeat.set(1, 1)
  texture.offset.set(0.25, 0)
  texture.minFilter = THREE.LinearFilter
  texture.magFilter = THREE.LinearFilter
  texture.generateMipmaps = false
  texture.needsUpdate = true
  return texture
}

export function normalizeTcc(value: number, usesPercentScale: boolean) {
  const normalized = usesPercentScale ? value / 100 : value
  return THREE.MathUtils.clamp(normalized, 0, 1)
}

export function calculateCloudAlpha(cloudCover: number) {
  const visibleCover = THREE.MathUtils.clamp(
    (cloudCover - CLOUD_MIN_VISIBLE_COVER) / (1 - CLOUD_MIN_VISIBLE_COVER),
    0,
    1,
  )
  return Math.pow(visibleCover, CLOUD_ALPHA_GAMMA)
}

function isPercentScale(frame: GlobeFrameAsset) {
  const unit = frame.unit.trim().toLowerCase()
  if (
    unit === '%' ||
    unit.includes('percent') ||
    unit.includes('процент')
  ) {
    return true
  }
  return frame.max !== undefined && frame.max > 1.5 && frame.max <= 100
}

function isMissingValue(value: number, frame: GlobeFrameAsset) {
  return (
    !Number.isFinite(value) ||
    (frame.noDataValue !== undefined && Object.is(value, frame.noDataValue))
  )
}
