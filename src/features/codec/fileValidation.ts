const MAX_BROWSER_UPLOAD_BYTES = 64 * 1024 ** 2
const SUPPORTED_SUFFIXES = ['.npz']

export function validateCodecFile(file: File): string | null {
  const normalizedName = file.name.trim().toLowerCase()

  if (!SUPPORTED_SUFFIXES.some((suffix) => normalizedName.endsWith(suffix))) {
    return 'Неподдерживаемый формат. Используйте NPZ с массивом data.'
  }
  if (file.size === 0) return 'Файл пуст.'
  if (file.size > MAX_BROWSER_UPLOAD_BYTES) {
    return 'Файл больше 64 МиБ. Подготовьте один ERA5-кадр в NPZ.'
  }
  return null
}

export function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} Б`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} КиБ`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} МиБ`
  return `${(bytes / 1024 ** 3).toFixed(2)} ГиБ`
}
