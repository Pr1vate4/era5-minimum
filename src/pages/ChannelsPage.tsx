import { getNavigationItem } from '../app/navigationConfig'
import { ChannelTable } from '../components/ChannelTable'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/common/PageHeader'
import { selectChannelMetrics } from '../data/resultsSelectors'
import { useResults } from '../hooks/useResults'

const page = getNavigationItem('channels')

export default function ChannelsPage() {
  const { data } = useResults()

  if (!data) {
    return <EmptyState title="Метрики недоступны" message="Результаты эксперимента ещё не загружены." />
  }

  const channels = selectChannelMetrics(data)

  return (
    <div className="space-y-5">
      <PageHeader title={page.title} description={page.description} />

      {channels.length < 28 ? (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-[12px] leading-5 text-amber-800">
          В текущем results.json присутствуют метрики {channels.length} из 28 каналов контракта ERA5.
          Отсутствующие каналы не создаются и не заполняются вымышленными значениями.
        </div>
      ) : null}

      {channels.length === 0 ? (
        <EmptyState
          title="Метрики каналов отсутствуют"
          message="Для страницы нужен непустой массив per_channel в results.json."
        />
      ) : (
        <ChannelTable channels={channels} />
      )}
    </div>
  )
}
