#!/usr/bin/env bash
# ============================================================
#  HealthBridge Claim Engine - start backend (:8000) + frontend (:5173)
#  Backend runs on Anaconda Python (has pandas/sklearn/xgboost).
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate Anaconda base if available (has the ML deps + fastapi/uvicorn).
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate base
fi

cleanup() {
  echo ""
  echo "[run] stopping servers..."
  kill "${API_PID:-}" "${WEB_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[run] starting FastAPI backend on http://localhost:8000 ..."
( cd "$ROOT" && python -m uvicorn api.main:app --reload --port 8000 ) &
API_PID=$!

echo "[run] starting Vite frontend on http://localhost:5173 ..."
( cd "$ROOT/web" && npm run dev ) &
WEB_PID=$!

echo ""
echo "  Backend : http://localhost:8000/api/health"
echo "  Frontend: http://localhost:5173"
echo "  Ctrl+C to stop both."
echo ""
wait
