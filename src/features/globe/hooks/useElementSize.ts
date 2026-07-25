import { useEffect, useRef, useState } from 'react'

export function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const element = ref.current
    if (!element) return

    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize((current) =>
        current.width === width && current.height === height ? current : { width, height },
      )
    })

    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  return { ref, ...size }
}
