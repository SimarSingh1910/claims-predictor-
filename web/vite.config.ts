import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server pinned to 5173 (strict) so the backend CORS allow-list and the
// run scripts always match. The backend base URL is http://localhost:8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
});
