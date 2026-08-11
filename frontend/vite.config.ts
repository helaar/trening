import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import tsconfigPaths from "vite-tsconfig-paths"

const backendUrl = process.env.VITE_BACKEND_URL ?? "http://localhost:5175"

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
  server: {
    host: true,
    proxy: {
      "/api": backendUrl,
      "/auth": backendUrl,
    },
  },
})
