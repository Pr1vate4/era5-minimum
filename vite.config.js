import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
var ignoredWatchDirectories = [
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
    var isPublicData = segments.some(function (segment, index) { return segment === 'data' && segments[index - 1] === 'public'; });
    return !isPublicData && segments.some(function (segment) { return ignoredWatchDirectories.indexOf(segment) >= 0; });
}
export default defineConfig(function (_a) {
    var mode = _a.mode;
    var env = loadEnv(mode, '.', '');
    return {
        plugins: [react()],
        server: {
            host: env.VITE_HOST || '0.0.0.0',
            port: Number(env.VITE_PORT || 5173),
            watch: {
                // Python environments and experiment artifacts can contain millions of
                // files. Keep public/data observable because it contains globe assets.
                ignored: shouldIgnoreWatchedPath,
            },
        },
        preview: {
            host: env.VITE_HOST || '0.0.0.0',
            port: Number(env.VITE_PORT || 5173),
        },
    };
});
