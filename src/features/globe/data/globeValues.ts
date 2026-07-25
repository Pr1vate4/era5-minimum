import { resolvePublicAssetUrl } from '../utils/resolvePublicAssetUrl'

const MAX_VALUES_CACHE_SIZE = 6
const valuesCache = new Map<string, Float32Array>()
const littleEndianHost = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1

export function decodeFloat32LittleEndian(buffer: ArrayBuffer) {
  if (buffer.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error('Размер values.bin не кратен четырём байтам.')
  }

  if (littleEndianHost) return new Float32Array(buffer)

  const view = new DataView(buffer)
  const values = new Float32Array(buffer.byteLength / Float32Array.BYTES_PER_ELEMENT)
  for (let index = 0; index < values.length; index += 1) {
    values[index] = view.getFloat32(index * Float32Array.BYTES_PER_ELEMENT, true)
  }
  return values
}

export function getCachedGlobeValues(valuesPath: string | undefined) {
  if (!valuesPath) return null
  const valuesUrl = resolvePublicAssetUrl(valuesPath)
  const cached = valuesCache.get(valuesUrl)
  if (!cached) return null

  touchValuesCache(valuesUrl, cached)
  return cached
}

export async function loadGlobeValues(
  valuesPath: string,
  expectedLength: number | undefined,
  signal: AbortSignal,
) {
  const valuesUrl = resolvePublicAssetUrl(valuesPath)
  const cached = valuesCache.get(valuesUrl)
  if (cached) {
    validateLength(cached, expectedLength)
    touchValuesCache(valuesUrl, cached)
    return cached
  }

  const response = await fetch(valuesUrl, { signal })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)

  const decoded = decodeFloat32LittleEndian(await response.arrayBuffer())
  validateLength(decoded, expectedLength)
  valuesCache.set(valuesUrl, decoded)
  pruneValuesCache(valuesUrl)
  return decoded
}

function validateLength(values: Float32Array, expectedLength: number | undefined) {
  if (expectedLength !== undefined && values.length !== expectedLength) {
    throw new Error(
      `values.bin содержит ${values.length} значений, ожидалось ${expectedLength}.`,
    )
  }
}

function touchValuesCache(url: string, values: Float32Array) {
  valuesCache.delete(url)
  valuesCache.set(url, values)
}

function pruneValuesCache(keepUrl: string) {
  while (valuesCache.size > MAX_VALUES_CACHE_SIZE) {
    const oldestUrl = valuesCache.keys().next().value as string | undefined
    if (!oldestUrl) return
    if (oldestUrl === keepUrl && valuesCache.size === 1) return
    valuesCache.delete(oldestUrl)
  }
}
