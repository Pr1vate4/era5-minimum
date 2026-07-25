import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
var ignoredRootDirectories = [
    '.venv',
    'venv',
    'data',
    'outputs',
    'checkpoints',
    'bitstreams',
    'artifacts',
    'submission',
];
function shouldIgnoreWatchedPath(path) {
    var segments = path.replace(/\\/g, '/').split('/').filter(Boolean);
    var isApplicationSource = segments.indexOf('src') >= 0;
    var isPublicAsset = segments.indexOf('public') >= 0;
    return (!isApplicationSource &&
        !isPublicAsset &&
        segments.some(function (segment) { return ignoredRootDirectories.indexOf(segment) >= 0; }));
}
export default defineConfig(function (_a) {
    var mode = _a.mode;
    var env = loadEnv(mode, '.', '');
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
    };
});
