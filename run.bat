@echo off
REM ============================================================
REM  HealthBridge Claim Engine - start backend (:8000) + frontend (:5173)
REM  Backend runs on Anaconda Python (has pandas/sklearn/xgboost).
REM ============================================================
setlocal

set ROOT=%~dp0

echo [run] starting FastAPI backend on http://localhost:8000 ...
start "healthbridge-api" cmd /k "cd /d %ROOT% && call conda activate base && python -m uvicorn api.main:app --reload --port 8000"

echo [run] starting Vite frontend on http://localhost:5173 ...
start "healthbridge-web" cmd /k "cd /d %ROOT%web && npm run dev"

echo.
echo   Backend : http://localhost:8000/api/health
echo   Frontend: http://localhost:5173
echo.
echo   Two terminal windows opened. Close them to stop the servers.
endlocal
