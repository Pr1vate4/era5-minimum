import { useMemo, useState } from 'react'
import type { Reconstruction } from '../types'

export function ReconstructionViewer({ reconstructions }: { reconstructions: Reconstruction[] }) {
  const uniqueChannels = Array.from(new Set(reconstructions.map((item) => item.channel)))
  const [channel, setChannel] = useState(uniqueChannels[0] ?? '')
  const [timestamp, setTimestamp] = useState(
    reconstructions.find((item) => item.channel === uniqueChannels[0])?.timestamp ?? '',
  )
  const [slide, setSlide] = useState(50)

  const visibleItems = useMemo(
    () => reconstructions.filter((item) => item.channel === channel),
    [channel, reconstructions],
  )

  const entry = useMemo(() => {
    return visibleItems.find((item) => item.timestamp === timestamp) ?? visibleItems[0] ?? reconstructions[0]
  }, [reconstructions, timestamp, visibleItems])

  if (!entry) {
    return null
  }

  return (
    <section id="reconstruction" className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-soft">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Visual check</p>
          <h2 className="text-2xl font-semibold text-white">Reconstruction viewer</h2>
        </div>
        <div className="flex gap-2">
          <select
            value={channel}
            onChange={(e) => {
              const next = e.target.value
              setChannel(next)
              const matchingTimestamp = reconstructions.find((item) => item.channel === next)?.timestamp ?? ''
              setTimestamp(matchingTimestamp)
            }}
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
          >
            {uniqueChannels.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
          <select
            value={timestamp}
            onChange={(e) => setTimestamp(e.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
          >
            {visibleItems.map((item) => (
              <option key={`${item.channel}-${item.timestamp}`} value={item.timestamp}>{item.timestamp}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {[
          ['Original', entry.original_image],
          ['Reconstructed', entry.reconstructed_image],
          ['Difference', entry.diff_image],
        ].map(([label, src]) => (
          <div key={label} className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
            <img src={src} alt={`${label} ${entry.channel}`} className="h-64 w-full object-cover" />
            <div className="px-3 py-2 text-sm text-slate-300">{label}</div>
          </div>
        ))}
      </div>

      <div className="mt-4">
        <div className="mb-2 flex items-center justify-between text-sm text-slate-300">
          <span>Before</span>
          <span>After</span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          value={slide}
          onChange={(e) => setSlide(Number(e.target.value))}
          className="w-full accent-sky-400"
        />
        <div className="mt-2 text-xs text-slate-400">Blend: {slide}% reconstructed / {(100 - slide)}% original</div>
      </div>
    </section>
  )
}
