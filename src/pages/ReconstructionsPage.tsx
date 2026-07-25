import { lazy, Suspense } from 'react'
import { getNavigationItem } from '../app/navigationConfig'
import { EmptyState } from '../components/EmptyState'
import { ReconstructionViewer } from '../components/ReconstructionViewer'
import { PageHeader } from '../components/common/PageHeader'
import { GlobeSkeleton } from '../features/globe/components/GlobeSkeleton'
import { selectReconstructions, toFiniteNumber } from '../data/resultsSelectors'
import { useResults } from '../hooks/useResults'

const page = getNavigationItem('reconstructions')
const AtmosphereGlobe = lazy(
  () => import('../features/globe/components/AtmosphereGlobe'),
)

export default function ReconstructionsPage() {
  const { data } = useResults()

  if (!data) {
    return <EmptyState title="Реконструкции недоступны" message="Результаты эксперимента ещё не загружены." />
  }

  const reconstructions = selectReconstructions(data)

  return (
    <div className="space-y-5">
      <PageHeader title={page.title} description={page.description} />
      <Suspense fallback={<GlobeSkeleton />}>
        <AtmosphereGlobe
          manifestUrl={data.globe?.manifestUrl}
          defaultChannel={data.globe?.defaultChannel}
          defaultTimestamp={data.globe?.defaultTimestamp}
          defaultGrid={data.meta.grid}
          availableModes={
            data.globe?.availableModes ?? [
              'original',
              'reconstruction',
              'absolute-error',
            ]
          }
          researchDefaults={{
            runId: data.meta.run_id,
            trainFrames: toFiniteNumber(data.meta.training_samples),
            compressionRatio: toFiniteNumber(data.compression.target_ratio),
            checkpoint: data.meta.checkpoint,
          }}
        />
      </Suspense>
      <ReconstructionViewer
        reconstructions={reconstructions}
        grid={data.meta.grid}
        compressionRatio={toFiniteNumber(data.compression.target_ratio)}
      />
    </div>
  )
}
