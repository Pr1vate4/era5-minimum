export function resolvePublicAssetUrl(path: string) {
  if (/^(?:https?:|data:|blob:)/i.test(path)) return path

  const base = import.meta.env.BASE_URL || '/'
  const normalizedBase = base.endsWith('/') ? base : `${base}/`
  const normalizedPath = path.replace(/^\.?\//, '')
  return `${normalizedBase}${normalizedPath}`
}
