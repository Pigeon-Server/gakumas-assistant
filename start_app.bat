@echo off
cd /d %~dp0
:: 检查是否有管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -Command "Start-Process '%~f0' -Verb RunAs -WorkingDirectory '%~dp0'"
    exit /b
)

.\.venv\Scripts\python.exe -m uvicorn app:app 
pause
