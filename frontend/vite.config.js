import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // 本地开发时把 /api 转发到本地跑着的 FastAPI 后端，
    // 跟生产环境 nginx 的 /api/ 反代规则保持一致，前端代码不用区分环境。
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
