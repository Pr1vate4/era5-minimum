import { ArrowLeft } from 'lucide-react'
import { Link } from 'react-router-dom'
import { EmptyState } from '../components/EmptyState'

export default function NotFoundPage() {
  return (
    <EmptyState
      title="Страница не найдена"
      message="Такого раздела нет в навигации МетеоКода."
      action={
        <Link
          to="/overview"
          className="ui-button-primary ui-focus-ring inline-flex h-9 items-center gap-2 rounded-lg px-3 text-sm font-semibold"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Вернуться к обзору
        </Link>
      }
    />
  )
}
