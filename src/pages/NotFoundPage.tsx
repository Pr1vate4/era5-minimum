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
          className="inline-flex h-9 items-center gap-2 rounded-lg bg-blue-600 px-3 text-sm font-semibold text-white hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Вернуться к обзору
        </Link>
      }
    />
  )
}
