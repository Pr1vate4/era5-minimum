import type { GlobeChannel } from '../types/globe'

export type GlobeParameterGroup = 'surface' | 'pressure'

export interface GlobeParameterConfig {
  channel: GlobeChannel
  group: GlobeParameterGroup
  name: string
  shortName: string
  description: string
  displayUnit: string
  fallbackRange: [number, number]
  colors: string[]
  scale: 'linear' | 'diverging' | 'logarithmic'
}

const temperatureColors = [
  '#172554',
  '#2563EB',
  '#22D3EE',
  '#10B981',
  '#FACC15',
  '#F97316',
  '#DC2626',
]

export const globeParameterConfigs: readonly GlobeParameterConfig[] = [
  {
    channel: 't2m',
    group: 'surface',
    name: 'Температура воздуха на высоте 2 м',
    shortName: 'Температура на 2 м',
    description: 'Распределение температуры воздуха на высоте 2 метра над поверхностью Земли.',
    displayUnit: 'K',
    fallbackRange: [220, 320],
    colors: temperatureColors,
    scale: 'linear',
  },
  {
    channel: 'mslp',
    group: 'surface',
    name: 'Давление на уровне моря',
    shortName: 'Давление на уровне моря',
    description: 'Атмосферное давление, приведённое к среднему уровню моря.',
    displayUnit: 'hPa',
    fallbackRange: [940, 1060],
    colors: ['#5B21B6', '#2563EB', '#BAE6FD', '#F8FAFC', '#FDBA74', '#DC2626'],
    scale: 'diverging',
  },
  {
    channel: 'u10',
    group: 'surface',
    name: 'Зональная компонента ветра на высоте 10 м',
    shortName: 'Зональный ветер на 10 м',
    description: 'Западно-восточная компонента скорости ветра на высоте 10 метров.',
    displayUnit: 'm/s',
    fallbackRange: [-30, 30],
    colors: ['#1E3A8A', '#38BDF8', '#F8FAFC', '#FDBA74', '#B91C1C'],
    scale: 'diverging',
  },
  {
    channel: 'v10',
    group: 'surface',
    name: 'Меридиональная компонента ветра на высоте 10 м',
    shortName: 'Меридиональный ветер на 10 м',
    description: 'Южно-северная компонента скорости ветра на высоте 10 метров.',
    displayUnit: 'm/s',
    fallbackRange: [-30, 30],
    colors: ['#1E3A8A', '#38BDF8', '#F8FAFC', '#FDBA74', '#B91C1C'],
    scale: 'diverging',
  },
  {
    channel: 'wind10',
    group: 'surface',
    name: 'Скорость ветра на высоте 10 м',
    shortName: 'Скорость ветра на 10 м',
    description: 'Модуль скорости ветра, рассчитанный по зональной и меридиональной компонентам.',
    displayUnit: 'm/s',
    fallbackRange: [0, 35],
    colors: ['#ECFEFF', '#67E8F9', '#22C55E', '#FACC15', '#F97316', '#B91C1C'],
    scale: 'linear',
  },
  {
    channel: 'tp6h',
    group: 'surface',
    name: 'Осадки за предыдущие 6 часов',
    shortName: 'Осадки за 6 часов',
    description: 'Суммарное количество осадков за шестичасовой интервал.',
    displayUnit: 'mm/6h',
    fallbackRange: [0, 50],
    colors: ['#F8FAFC', '#BAE6FD', '#38BDF8', '#2563EB', '#7E22CE', '#BE123C'],
    scale: 'logarithmic',
  },
  {
    channel: 'sst',
    group: 'surface',
    name: 'Температура поверхности моря',
    shortName: 'Температура моря',
    description: 'Температура верхнего слоя океана; суша может быть отмечена как отсутствие данных.',
    displayUnit: 'K',
    fallbackRange: [270, 310],
    colors: temperatureColors,
    scale: 'linear',
  },
  {
    channel: 'tcwv',
    group: 'surface',
    name: 'Полное содержание водяного пара',
    shortName: 'Водяной пар',
    description: 'Интегральное содержание водяного пара во всей атмосферной колонне.',
    displayUnit: 'kg/m²',
    fallbackRange: [0, 70],
    colors: ['#F0FDFA', '#99F6E4', '#2DD4BF', '#0EA5E9', '#1D4ED8', '#312E81'],
    scale: 'linear',
  },
  {
    channel: 'tcc',
    group: 'surface',
    name: 'Общая облачность',
    shortName: 'Общая облачность',
    description: 'Доля небосвода, закрытая облаками, от 0 до 1.',
    displayUnit: 'доля',
    fallbackRange: [0, 1],
    colors: ['#334155', '#64748B', '#94A3B8', '#CBD5E1', '#F8FAFC'],
    scale: 'linear',
  },
  {
    channel: 'T',
    group: 'pressure',
    name: 'Температура атмосферы',
    shortName: 'Температура',
    description: 'Температура воздуха на выбранном изобарическом уровне.',
    displayUnit: 'K',
    fallbackRange: [220, 310],
    colors: temperatureColors,
    scale: 'linear',
  },
  {
    channel: 'U',
    group: 'pressure',
    name: 'Зональная компонента ветра',
    shortName: 'Зональный ветер',
    description: 'Западно-восточная компонента ветра на выбранном изобарическом уровне.',
    displayUnit: 'm/s',
    fallbackRange: [-60, 60],
    colors: ['#172554', '#2563EB', '#BAE6FD', '#F8FAFC', '#FDBA74', '#DC2626', '#7F1D1D'],
    scale: 'diverging',
  },
  {
    channel: 'V',
    group: 'pressure',
    name: 'Меридиональная компонента ветра',
    shortName: 'Меридиональный ветер',
    description: 'Южно-северная компонента ветра на выбранном изобарическом уровне.',
    displayUnit: 'm/s',
    fallbackRange: [-50, 50],
    colors: ['#172554', '#2563EB', '#BAE6FD', '#F8FAFC', '#FDBA74', '#DC2626', '#7F1D1D'],
    scale: 'diverging',
  },
  {
    channel: 'Z',
    group: 'pressure',
    name: 'Геопотенциал',
    shortName: 'Геопотенциал',
    description: 'Геопотенциал на выбранном изобарическом уровне атмосферы.',
    displayUnit: 'm²/s²',
    fallbackRange: [0, 60000],
    colors: ['#0F766E', '#22C55E', '#A3E635', '#FACC15', '#F97316', '#991B1B'],
    scale: 'linear',
  },
  {
    channel: 'Q',
    group: 'pressure',
    name: 'Удельная влажность',
    shortName: 'Удельная влажность',
    description: 'Массовая доля водяного пара на выбранном изобарическом уровне.',
    displayUnit: 'kg/kg',
    fallbackRange: [0, 0.02],
    colors: ['#F0FDFA', '#99F6E4', '#2DD4BF', '#0EA5E9', '#1D4ED8', '#312E81'],
    scale: 'linear',
  },
]

const configByChannel = new Map(
  globeParameterConfigs.map((configuration) => [configuration.channel, configuration]),
)

export function getGlobeParameterConfig(channel: GlobeChannel) {
  return configByChannel.get(channel) ?? globeParameterConfigs[0]
}

export function isPressureChannel(channel: GlobeChannel) {
  return getGlobeParameterConfig(channel).group === 'pressure'
}

export function toDisplayValue(channel: GlobeChannel, value: number, sourceUnit?: string) {
  if (channel === 'mslp' && sourceUnit?.toLowerCase() === 'pa') return value / 100
  if (channel === 'tp6h' && sourceUnit?.toLowerCase() === 'm') return value * 1000
  return value
}

export function getDisplayUnit(channel: GlobeChannel, sourceUnit?: string) {
  if (channel === 'mslp') return 'hPa'
  if (channel === 'tp6h') return 'mm/6h'
  return sourceUnit || getGlobeParameterConfig(channel).displayUnit
}
