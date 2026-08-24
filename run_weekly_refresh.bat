@echo off
cd /d "%~dp0"

".venv\Scripts\python.exe" evaluate_recommendations.py
if %ERRORLEVEL% NEQ 0 (
    echo evaluate_recommendations.py failed with exit code %ERRORLEVEL% - skipping build.
    exit /b %ERRORLEVEL%
)

".venv\Scripts\python.exe" build_recommendation_matrices.py
exit /b %ERRORLEVEL%
