import { useCallback, useEffect, useState } from 'react'
import { parseGlobeManifest } from '../data/globeSelectors'
import type { GlobeManifestState } from '../types/globe'
import { resolvePublicAssetUrl } from '../utils/resolvePublicAssetUrl'

const manifestCache = new Map<string, ReturnType<typeof parseGlobeManifest>>()

export function useGlobeManifest(manifestPath: string): GlobeManifestState {
  const manifestUrl = resolvePublicAssetUrl(manifestPath)
  const cached = manifestCache.get(manifestUrl)
  const [manifest, setManifest] = useState(cached?.manifest ?? null)
  const [invalidFrameCount, setInvalidFrameCount] = useState(cached?.invalidFrameCount ?? 0)
  const [loading, setLoading] = useState(!cached)
  const [error, setError] = useState<string | null>(null)
  const [requestVersion, setRequestVersion] = useState(0)

  const reload = useCallback(() => {
    manifestCache.delete(manifestUrl)
    setRequestVersion((version) => version + 1)
  }, [manifestUrl])

  useEffect(() => {
    const cachedManifest = manifestCache.get(manifestUrl)
    if (cachedManifest) {
      setManifest(cachedManifest.manifest)
      setInvalidFrameCount(cachedManifest.invalidFrameCount)
      setLoading(false)
      setError(null)
      return
    }

    const controller = new AbortController()
    setLoading(true)
    setError(null)

    fetch(manifestUrl, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json() as Promise<unknown>
      })
      .then((json) => {
        const parsed = parseGlobeManifest(json)
        manifestCache.set(manifestUrl, parsed)
        setManifest(parsed.manifest)
        setInvalidFrameCount(parsed.invalidFrameCount)
        setLoading(false)

        if (parsed.invalidFrameCount > 0 && import.meta.env.DEV) {
          console.warn(`Globe manifest: пропущено некорректных записей: ${parsed.invalidFrameCount}`)
        }
      })
      .catch((caughtError: unknown) => {
        if (caughtError instanceof DOMException && caughtError.name === 'AbortError') return
        setManifest(null)
        setLoading(false)
        setError(
          caughtError instanceof Error
            ? caughtError.message
            : 'Неизвестная ошибка загрузки manifest.',
        )
      })

    return () => controller.abort()
  }, [manifestUrl, requestVersion])

  return { manifest, loading, error, invalidFrameCount, reload }
}
