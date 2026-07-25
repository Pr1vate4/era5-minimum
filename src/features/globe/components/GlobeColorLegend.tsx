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
      className="border-b border-[#E4E7EC] bg-white px-5 py-4 lg:px-6"
      aria-label={`Шкала значений от ${formatLegendValue(scale.min)} до ${formatLegendValue(scale.max)} ${scale.unit}`}
    >
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="text-[13px] font-semibold text-[#344054]">
          Шкала значений
        </span>
        <span className="text-[13px] font-semibold text-[#475467]">{scale.unit}</span>
      </div>
      <div
        className="h-3 rounded-full border border-black/5 shadow-inner"
        style={{ background: `linear-gradient(90deg, ${scale.colors.join(', ')})` }}
        aria-hidden="true"
      />
      <div className="mt-2 grid grid-cols-7 gap-1">
        {ticks.map((tick, index) => (
          <span
            key={`${tick}-${index}`}
            className={`text-[12px] font-medium tabular-nums text-[#667085] ${
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
