import { readFileSync } from 'node:fs'
import { dirname, isAbsolute, resolve } from 'node:path'
import { homedir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const settingsFile = process.env.DNATRAINER_SETTINGS ?? resolve(dirname(fileURLToPath(import.meta.url)), '../backend/settings.json')
const settings = JSON.parse(readFileSync(settingsFile, 'utf8'))
const projectRoot = resolve(dirname(settingsFile), '..')
const configuredStaticDir = settings.frontend_static_dir ?? 'TrainerView/dist'
const staticDir = configuredStaticDir.startsWith('~/')
  ? resolve(homedir(), configuredStaticDir.slice(2)) : configuredStaticDir
const backendHost = ['0.0.0.0', '::'].includes(settings.backend_host ?? '0.0.0.0')
  ? '127.0.0.1' : settings.backend_host
const proxyHost = backendHost.includes(':') ? `[${backendHost}]` : backendHost

export default defineConfig({
  plugins: [react({ babel: { plugins: [['babel-plugin-react-compiler']] } })],
  build: {
    outDir: staticDir ? (isAbsolute(staticDir) ? staticDir : resolve(projectRoot, staticDir)) : 'dist',
  },
  server: {
    port: settings.frontend_dev_port ?? 5173,
    host: settings.frontend_dev_host ?? '0.0.0.0',
    allowedHosts: settings.frontend_allowed_hosts ?? ['localhost'],
    proxy: { '/api': `http://${proxyHost}:${settings.backend_port ?? 25565}` },
  },
})
