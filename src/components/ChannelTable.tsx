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

const groupLabel = {
  surface: 'Приземные поля',
  pressure: 'Атмосферные поля',
}

export function ChannelTable({ channels }: { channels: PerChannel[] }) {
  const [group, setGroup] = useState<'surface' | 'pressure' | 'all'>('all')

  const filtered = useMemo(() => {
    const next = group === 'all' ? channels : channels.filter((item) => item.group === group)
    return next.sort((a, b) => a.nrmse - b.nrmse)
  }, [channels, group])

  return (
    <section id="per-channel" className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between gap-4">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Метрики по каналам</div>
          <h2 className="mt-1 text-[17px] font-semibold text-slate-900">Качество по каналам</h2>
        </div>
        <div className="flex gap-2">
          {(['all', 'surface', 'pressure'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setGroup(tab)}
              className={`rounded-lg border px-3 py-1.5 text-[12px] transition ${
                group === tab ? 'border-blue-200 bg-blue-50 text-blue-700' : 'border-slate-200 text-slate-600 hover:border-slate-300'
              }`}
            >
              {tab === 'all' ? 'Все' : groupLabel[tab]}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-slate-200">
        <table className="min-w-full text-[12px]">
          <thead className="bg-slate-50 text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">Канал</th>
              <th className="px-3 py-2 text-left">Группа</th>
              <th className="px-3 py-2 text-left">RMSE</th>
              <th className="px-3 py-2 text-left">NRMSE</th>
              <th className="px-3 py-2 text-left">PSNR</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((item) => (
              <tr key={item.code} className="border-t border-slate-200 bg-white hover:bg-slate-50/80">
                <td className="px-3 py-2 text-slate-900">{item.code}</td>
                <td className="px-3 py-2"><span className={`rounded-full border px-2 py-1 text-[11px] ${groupAccent[item.group]}`}>{groupLabel[item.group]}</span></td>
                <td className="px-3 py-2 text-slate-700">{item.rmse.toFixed(2)} {item.unit}</td>
                <td className="px-3 py-2"><span className={`rounded-full px-2 py-1 ${heatColor(item.nrmse)}`}>{item.nrmse.toFixed(3)}</span></td>
                <td className="px-3 py-2 text-slate-700">{item.psnr.toFixed(1)} dB</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
