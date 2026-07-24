import { useMemo, useState } from 'react'
import type { PerChannel } from '../types'

const groupAccent = {
  surface: 'text-cyan-300 border-cyan-500/30 bg-cyan-500/10',
  pressure: 'text-violet-300 border-violet-500/30 bg-violet-500/10',
}

const heatColor = (value: number) => {
  if (value < 0.035) return 'bg-emerald-500/20 text-emerald-200'
  if (value < 0.045) return 'bg-sky-500/20 text-sky-200'
  if (value < 0.055) return 'bg-amber-500/20 text-amber-100'
  return 'bg-rose-500/20 text-rose-200'
}

export function ChannelTable({ channels }: { channels: PerChannel[] }) {
  const [group, setGroup] = useState<'surface' | 'pressure' | 'all'>('all')

  const filtered = useMemo(() => {
    const next = group === 'all' ? channels : channels.filter((item) => item.group === group)
    return next.sort((a, b) => a.nrmse - b.nrmse)
  }, [channels, group])

  return (
    <section id="per-channel" className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-soft">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Per-channel quality</p>
          <h2 className="text-2xl font-semibold text-white">Channel metrics</h2>
        </div>
        <div className="flex gap-2">
          {(['all', 'surface', 'pressure'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setGroup(tab)}
              className={`rounded-full border px-3 py-1 text-sm ${
                group === tab ? 'border-accent bg-accent/20 text-white' : 'border-slate-700 text-slate-300'
              }`}
            >
              {tab === 'all' ? 'All' : tab}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-slate-800">
        <table className="min-w-full divide-y divide-slate-800 text-sm">
          <thead className="bg-slate-950/70 text-slate-300">
            <tr>
              <th className="px-4 py-3 text-left">Channel</th>
              <th className="px-4 py-3 text-left">Group</th>
              <th className="px-4 py-3 text-left">RMSE</th>
              <th className="px-4 py-3 text-left">NRMSE</th>
              <th className="px-4 py-3 text-left">PSNR</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((item) => (
              <tr key={item.code} className="border-t border-slate-800 bg-slate-900/30">
                <td className="px-4 py-3 text-white">{item.name}</td>
                <td className="px-4 py-3">
                  <span className={`rounded-full border px-2 py-1 text-xs ${groupAccent[item.group]}`}>
                    {item.group}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-200">{item.rmse.toFixed(2)} {item.unit}</td>
                <td className="px-4 py-3">
                  <span className={`rounded-full px-2 py-1 ${heatColor(item.nrmse)}`}>
                    {item.nrmse.toFixed(3)}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-200">{item.psnr.toFixed(1)} dB</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
