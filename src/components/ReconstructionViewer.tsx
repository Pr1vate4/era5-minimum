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
    <section id="reconstruction" className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between gap-4">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Реконструкции</div>
          <h2 className="mt-1 text-[17px] font-semibold text-slate-900">Просмотр реконструкции</h2>
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
            className="rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-[12px] text-slate-700"
          >
            {uniqueChannels.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
          <select
            value={timestamp}
            onChange={(e) => setTimestamp(e.target.value)}
            className="rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-[12px] text-slate-700"
          >
            {visibleItems.map((item) => (
              <option key={`${item.channel}-${item.timestamp}`} value={item.timestamp}>{item.timestamp}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {[
          ['Оригинал', entry.original_image],
          ['Восстановление', entry.reconstructed_image],
          ['Разница', entry.diff_image],
        ].map(([label, src]) => (
          <div key={label} className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
            <img src={src} alt={`${label} ${entry.channel}`} className="h-[180px] w-full object-cover" />
            <div className="px-3 py-2 text-[12px] text-slate-600">{label}</div>
          </div>
        ))}
      </div>

      <div className="mt-3">
        <div className="mb-1 flex items-center justify-between text-[12px] text-slate-600">
          <span>До</span>
          <span>После</span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          value={slide}
          onChange={(e) => setSlide(Number(e.target.value))}
          className="w-full accent-blue-500"
        />
        <div className="mt-1 text-[11px] text-slate-500">Переместите ползунок для сравнения: {slide}% восстановлено / {(100 - slide)}% исходное</div>
      </div>
    </section>
  )
}
