import {
  BadgeCheck,
  BrainCircuit,
  Gauge,
  Images,
  LayoutDashboard,
  LineChart,
  TableProperties,
  TrendingUp,
  Waves,
  type LucideIcon,
} from 'lucide-react'

export interface NavigationItem {
  id: string
  path: string
  title: string
  shortTitle: string
  description: string
  icon: LucideIcon
}

export const navigationItems: NavigationItem[] = [
  {
    id: 'overview',
    path: '/overview',
    title: 'Обзор эксперимента',
    shortTitle: 'Обзор',
    description: 'Главные результаты запуска и итоговый статус модели',
    icon: LayoutDashboard,
  },
  {
    id: 'criteria',
    path: '/criteria',
    title: 'Критерии допуска',
    shortTitle: 'Критерии',
    description: 'Проверка обязательных требований к качеству и bitstream',
    icon: BadgeCheck,
  },
  {
    id: 'data-efficiency',
    path: '/data-efficiency',
    title: 'Эффективность обучающей выборки',
    shortTitle: 'Выборка',
    description: 'Зависимость качества от количества уникальных погодных кадров',
    icon: TrendingUp,
  },
  {
    id: 'rate-distortion',
    path: '/rate-distortion',
    title: 'Сжатие и качество',
    shortTitle: 'Сжатие',
    description: 'Компромисс между размером bitstream и качеством реконструкции',
    icon: LineChart,
  },
  {
    id: 'channels',
    path: '/channels',
    title: 'Метрики по каналам',
    shortTitle: 'Каналы',
    description: 'RMSE, NRMSE и PSNR доступных погодных каналов',
    icon: TableProperties,
  },
  {
    id: 'reconstructions',
    path: '/reconstructions',
    title: 'Просмотр реконструкций',
    shortTitle: 'Реконструкции',
    description: 'Сравнение исходных, восстановленных полей и карт ошибки',
    icon: Images,
  },
  {
    id: 'spectral',
    path: '/spectral',
    title: 'Спектральный анализ',
    shortTitle: 'Спектр',
    description: 'Проверка сохранения пространственных структур',
    icon: Waves,
  },
  {
    id: 'probe',
    path: '/probe',
    title: 'Latent-прогноз на +6 часов',
    shortTitle: 'Прогноз',
    description: 'Проверка полезности latent-представления для прогноза',
    icon: BrainCircuit,
  },
  {
    id: 'resources',
    path: '/resources',
    title: 'Ресурсы и ограничения',
    shortTitle: 'Ресурсы',
    description: 'Использование GPU, VRAM и compute-бюджета',
    icon: Gauge,
  },
]

export function findNavigationItem(pathname: string) {
  return navigationItems.find((item) => item.path === pathname)
}

export function getNavigationItem(id: string) {
  const item = navigationItems.find((entry) => entry.id === id)

  if (!item) {
    throw new Error(`Unknown navigation item: ${id}`)
  }

  return item
}
