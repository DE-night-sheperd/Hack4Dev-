import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import cesiumPlugin from 'vite-plugin-cesium'

const cesium = (cesiumPlugin as any).default ?? cesiumPlugin

export default defineConfig({
  plugins: [react(), cesium()],
})