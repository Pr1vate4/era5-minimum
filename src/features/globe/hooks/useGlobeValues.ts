import { useEffect, useState } from 'react'
import { getCachedGlobeValues, loadGlobeValues } from '../data/globeValues'

export function useGlobeValues(valuesPath: string | undefined, expectedLength: number | undefined) {
  const [values, setValues] = useState<Float32Array | null>(
    getCachedGlobeValues(valuesPath),
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!valuesPath) {
      setValues(null)
      setLoading(false)
      setError(null)
      return
    }

    const cached = getCachedGlobeValues(valuesPath)
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

    void loadGlobeValues(valuesPath, expectedLength, controller.signal)
      .then((loadedValues) => {
        setValues(loadedValues)
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
  }, [expectedLength, valuesPath])

  return { values, loading, error }
}
