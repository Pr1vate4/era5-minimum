import { useEffect, useState } from 'react'

const STORAGE_KEY = 'meteokod.globe.autoRotate'

function prefersReducedMotion() {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

export function useGlobeControls() {
  const [reducedMotion, setReducedMotion] = useState(prefersReducedMotion)
  const [autoRotate, setAutoRotate] = useState(() => {
    if (typeof window === 'undefined' || prefersReducedMotion()) return false
    return window.localStorage.getItem(STORAGE_KEY) !== 'false'
  })
  const [resetSignal, setResetSignal] = useState(0)

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const handleChange = () => {
      setReducedMotion(media.matches)
      if (media.matches) setAutoRotate(false)
    }

    media.addEventListener('change', handleChange)
    return () => media.removeEventListener('change', handleChange)
  }, [])

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, String(autoRotate))
  }, [autoRotate])

  return {
    autoRotate,
    setAutoRotate,
    reducedMotion,
    resetSignal,
    resetView: () => setResetSignal((signal) => signal + 1),
  }
}
