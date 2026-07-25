import { buildLegendTicks, formatLegendValue, getGlobeColorScale } from '../data/globeColorScales'
import type { GlobeFrameAsset, GlobeMode } from '../types/globe'

export function GlobeColorLegend({
  frame,
  mode,
}: {
  frame: GlobeFrameAsset | undefined
  mode: GlobeMode
}) {
  const scale = getGlobeColorScale(frame, mode)
  const ticks = buildLegendTicks(scale.min, scale.max)

  return (
    <div
      className="border-t border-[#E4E7EC] px-4 py-4 sm:px-5"
      aria-label={`Шкала значений от ${formatLegendValue(scale.min)} до ${formatLegendValue(scale.max)} ${scale.unit}`}
    >
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">
          Шкала значений
        </span>
        <span className="text-[11px] font-semibold text-slate-600">{scale.unit}</span>
      </div>
      <div
        className="h-2.5 rounded-full border border-black/5 shadow-inner"
        style={{ background: `linear-gradient(90deg, ${scale.colors.join(', ')})` }}
        aria-hidden="true"
      />
      <div className="mt-1.5 grid grid-cols-7 gap-1">
        {ticks.map((tick, index) => (
          <span
            key={`${tick}-${index}`}
            className={`text-[9px] font-medium text-slate-500 sm:text-[10px] ${
              index === 0 ? 'text-left' : index === ticks.length - 1 ? 'text-right' : 'text-center'
            }`}
          >
            {formatLegendValue(tick)}
          </span>
        ))}
      </div>
    </div>
  )
}
