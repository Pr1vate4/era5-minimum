import { getNavigationItem } from '../app/navigationConfig'
import { EmptyState } from '../components/EmptyState'
import { ReconstructionViewer } from '../components/ReconstructionViewer'
import { PageHeader } from '../components/common/PageHeader'
import { selectReconstructions, toFiniteNumber } from '../data/resultsSelectors'
import { useResults } from '../hooks/useResults'

const page = getNavigationItem('reconstructions')

export default function ReconstructionsPage() {
  const { data } = useResults()

  if (!data) {
    return <EmptyState title="Реконструкции недоступны" message="Результаты эксперимента ещё не загружены." />
  }

  const reconstructions = selectReconstructions(data)

  return (
    <div className="space-y-5">
      <PageHeader title={page.title} description={page.description} />
      <ReconstructionViewer
        reconstructions={reconstructions}
        grid={data.meta.grid}
        compressionRatio={toFiniteNumber(data.compression.target_ratio)}
      />
    </div>
  )
}
