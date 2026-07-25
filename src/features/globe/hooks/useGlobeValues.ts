import { useEffect, useState } from 'react'
import { resolvePublicAssetUrl } from '../utils/resolvePublicAssetUrl'

const valuesCache = new Map<string, Float32Array>()
const littleEndianHost = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1

function decodeFloat32LittleEndian(buffer: ArrayBuffer) {
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

export function useGlobeValues(valuesPath: string | undefined, expectedLength: number | undefined) {
  const valuesUrl = valuesPath ? resolvePublicAssetUrl(valuesPath) : undefined
  const [values, setValues] = useState<Float32Array | null>(
    valuesUrl ? (valuesCache.get(valuesUrl) ?? null) : null,
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!valuesUrl) {
      setValues(null)
      setLoading(false)
      setError(null)
      return
    }

    const cached = valuesCache.get(valuesUrl)
    if (cached) {
      setValues(cached)
      setLoading(false)
      setError(null)
      return
    }

    const controller = new AbortController()
    setValues(null)
    setLoading(true)
    setError(null)

    fetch(valuesUrl, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.arrayBuffer()
      })
      .then((buffer) => {
        const decoded = decodeFloat32LittleEndian(buffer)
        if (expectedLength !== undefined && decoded.length !== expectedLength) {
          throw new Error(
            `values.bin содержит ${decoded.length} значений, ожидалось ${expectedLength}.`,
          )
        }
        valuesCache.set(valuesUrl, decoded)
        setValues(decoded)
        setLoading(false)
      })
      .catch((caughtError: unknown) => {
        if (caughtError instanceof DOMException && caughtError.name === 'AbortError') return
        setLoading(false)
        setError(
          caughtError instanceof Error
            ? caughtError.message
            : 'Неизвестная ошибка загрузки числового слоя.',
        )
      })

    return () => controller.abort()
  }, [expectedLength, valuesUrl])

  return { values, loading, error }
}
