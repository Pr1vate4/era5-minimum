import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

const ignoredRootDirectories: string[] = [
  '.venv',
  'venv',
  'data',
  'outputs',
  'checkpoints',
  'bitstreams',
  'artifacts',
  'submission',
]

function shouldIgnoreWatchedPath(path: string) {
  const segments = path.replace(/\\/g, '/').split('/').filter(Boolean)
  const isApplicationSource = segments.indexOf('src') >= 0
  const isPublicAsset = segments.indexOf('public') >= 0

  return (
    !isApplicationSource &&
    !isPublicAsset &&
    segments.some((segment) => ignoredRootDirectories.indexOf(segment) >= 0)
  )
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')

  return {
    plugins: [react()],
    server: {
      host: env.VITE_HOST || '0.0.0.0',
      port: Number(env.VITE_PORT || 5173),
      proxy: {
        '/api': {
          target: env.VITE_WEATHER_API_TARGET || 'http://localhost:8000',
          changeOrigin: true,
        },
      },
      watch: {
        // Python environments and experiment artifacts can contain millions of
        // files. Ignore only root artifact directories so source folders named
        // "data" still trigger HMR.
        ignored: shouldIgnoreWatchedPath,
      },
    },
    preview: {
      host: env.VITE_HOST || '0.0.0.0',
      port: Number(env.VITE_PORT || 5173),
    },
  }
})
