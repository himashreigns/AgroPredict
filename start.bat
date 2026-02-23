@echo off
echo ============================================================
echo  SL Produce Price Predictor - Starting Services
echo ============================================================
echo.
echo [1/2] Starting FastAPI backend on http://localhost:8000
start "FastAPI Backend" cmd /k "cd /d d:\Mora\Academic\Sem 7\ML\Final\Veg && python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo [2/2] Starting React frontend on http://localhost:5173
start "React Frontend" cmd /k "cd /d d:\Mora\Academic\Sem 7\ML\Final\Veg\frontend && npm run dev"

echo.
echo ============================================================
echo  App will be available at: http://localhost:5173
echo  API docs at:             http://localhost:8000/docs
echo ============================================================
echo.
timeout /t 5 /nobreak >nul
start "" "http://localhost:5173"
