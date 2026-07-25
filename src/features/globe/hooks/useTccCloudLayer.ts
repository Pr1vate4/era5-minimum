import { useCallback, useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { createTccCloudTexture } from '../data/tccCloudTexture'
import { loadGlobeValues } from '../data/globeValues'
import type { GlobeFrameAsset } from '../types/globe'

export type TccCloudTextureResult = {
  texture: THREE.DataTexture
  width: number
  height: number
  timestamp: string
  grid: GlobeFrameAsset['grid']
  frame: GlobeFrameAsset
  cacheKey: string
}

type TccCloudLayerState = {
  result: TccCloudTextureResult | null
  loading: boolean
  error: string | null
}

const MAX_CLOUD_TEXTURE_CACHE_SIZE = 5
const cloudTextureCache = new Map<string, TccCloudTextureResult>()
const activeCloudTextureKeys = new Set<string>()
let cloudLayerConsumers = 0
let disposeTimer: number | undefined

export function useTccCloudLayer(
  frame: GlobeFrameAsset | undefined,
  enabled: boolean,
  liveValues?: Float32Array,
  liveLoading = false,
  liveError: string | null = null,
) {
  const [revision, setRevision] = useState(0)
  const requestIdRef = useRef(0)
  const [state, setState] = useState<TccCloudLayerState>({
    result: null,
    loading: false,
    error: null,
  })

  useEffect(() => {
    cloudLayerConsumers += 1
    if (disposeTimer !== undefined) {
      window.clearTimeout(disposeTimer)
      disposeTimer = undefined
    }

    return () => {
      cloudLayerConsumers -= 1
      if (cloudLayerConsumers > 0) return
      disposeTimer = window.setTimeout(() => {
        if (cloudLayerConsumers > 0) return
        cloudTextureCache.forEach(({ texture }) => texture.dispose())
        cloudTextureCache.clear()
        activeCloudTextureKeys.clear()
      }, 500)
    }
  }, [])

  useEffect(() => {
    const cacheKey = state.result?.cacheKey
    if (!cacheKey) return

    activeCloudTextureKeys.add(cacheKey)
    return () => {
      window.setTimeout(() => {
        activeCloudTextureKeys.delete(cacheKey)
        pruneCloudTextureCache()
      }, 500)
    }
  }, [state.result?.cacheKey])

  useEffect(() => {
    const requestId = requestIdRef.current + 1
    requestIdRef.current = requestId

    if (!enabled) {
      setState({ result: null, loading: false, error: null })
      return
    }
    if (!frame) {
      setState({ result: null, loading: false, error: null })
      return
    }
    if (frame.apiVariable) {
      if (liveLoading) {
        setState({ result: null, loading: true, error: null })
        return
      }
      if (liveError) {
        setState({ result: null, loading: false, error: liveError })
        return
      }
      if (!liveValues) {
        setState({
          result: null,
          loading: true,
          error: null,
        })
        return
      }

      const cacheKey = getCloudCacheKey(frame)
      const cached = cloudTextureCache.get(cacheKey)
      if (cached) {
        touchCloudTextureCache(cacheKey, cached)
        setState({ result: cached, loading: false, error: null })
        return
      }

      const texture = createTccCloudTexture({ frame, values: liveValues })
      const result: TccCloudTextureResult = {
        texture,
        width: frame.width,
        height: frame.height,
        timestamp: frame.timestamp,
        grid: frame.grid,
        frame,
        cacheKey,
      }
      cloudTextureCache.set(cacheKey, result)
      pruneCloudTextureCache(cacheKey)
      setState({ result, loading: false, error: null })
      return
    }
    if (!frame.valuesUrl) {
      setState({
        result: null,
        loading: false,
        error: 'Для выбранного кадра TCC отсутствует values.bin.',
      })
      return
    }

    const cacheKey = getCloudCacheKey(frame)
    const cached = cloudTextureCache.get(cacheKey)
    if (cached) {
      touchCloudTextureCache(cacheKey, cached)
      setState({ result: cached, loading: false, error: null })
      return
    }

    const controller = new AbortController()
    setState((current) => ({ ...current, loading: true, error: null }))

    void loadGlobeValues(
      frame.valuesUrl,
      frame.width * frame.height,
      controller.signal,
    )
      .then((values) => {
        if (controller.signal.aborted || requestId !== requestIdRef.current) return

        const texture = createTccCloudTexture({ frame, values })
        const result: TccCloudTextureResult = {
          texture,
          width: frame.width,
          height: frame.height,
          timestamp: frame.timestamp,
          grid: frame.grid,
          frame,
          cacheKey,
        }
        cloudTextureCache.set(cacheKey, result)
        pruneCloudTextureCache(cacheKey)
        setState({ result, loading: false, error: null })
      })
      .catch((caughtError: unknown) => {
        if (controller.signal.aborted || requestId !== requestIdRef.current) return
        setState({
          result: null,
          loading: false,
          error:
            caughtError instanceof Error
              ? caughtError.message
              : 'Неизвестная ошибка загрузки облачности TCC.',
        })
      })

    return () => controller.abort()
  }, [enabled, frame, liveError, liveLoading, liveValues, revision])

  const retry = useCallback(() => {
    setRevision((current) => current + 1)
  }, [])

  return {
    texture: state.result?.texture,
    loadedFrame: state.result?.frame,
    loading: state.loading,
    error: state.error,
    retry,
  }
}

function getCloudCacheKey(frame: GlobeFrameAsset) {
  return `${frame.mode}:${frame.channel}:${frame.timestamp}:${frame.grid}:${frame.id}`
}

function touchCloudTextureCache(key: string, result: TccCloudTextureResult) {
  cloudTextureCache.delete(key)
  cloudTextureCache.set(key, result)
}

function pruneCloudTextureCache(keepKey?: string) {
  if (cloudTextureCache.size <= MAX_CLOUD_TEXTURE_CACHE_SIZE) return

  for (const [key, result] of cloudTextureCache) {
    if (cloudTextureCache.size <= MAX_CLOUD_TEXTURE_CACHE_SIZE) return
    if (key === keepKey || activeCloudTextureKeys.has(key)) continue
    result.texture.dispose()
    cloudTextureCache.delete(key)
  }
}
