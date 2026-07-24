import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')

  return {
    plugins: [react()],
    server: {
      host: env.VITE_HOST || '0.0.0.0',
      port: Number(env.VITE_PORT || 5173),
      watch: {
        // Python environments and experiment artifacts can contain millions of
        // files. They are not frontend sources and exhaust Linux inotify limits.
        ignored: [
          '**/.venv/**',
          '**/venv/**',
          '**/data/**',
          '**/outputs/**',
          '**/checkpoints/**',
          '**/bitstreams/**',
          '**/artifacts/**',
          '**/submission/**',
        ],
      },
    },
    preview: {
      host: env.VITE_HOST || '0.0.0.0',
      port: Number(env.VITE_PORT || 5173),
    },
  }
})
