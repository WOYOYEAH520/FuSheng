@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PORT=7860
set PYTHONUTF8=1

netstat -ano | findstr /C:":%PORT% " | findstr /C:"LISTENING" >nul 2>&1
if not errorlevel 1 goto already

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] 缺少运行环境 .venv
    echo         请先双击 setup.bat，装好依赖再运行本文件。
    pause
    exit /b 1
)

echo ==============================================
echo   浮声 / 翻唱工作台
echo   Address : http://127.0.0.1:%PORT%
echo   Stop    : close this window
echo ==============================================
echo.

start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:%PORT%'"

".venv\Scripts\python.exe" "webapp\app.py"

echo.
echo Server stopped. Press any key to close.
pause >nul
exit /b 0

:already
echo Server already running - opening browser ...
start "" http://127.0.0.1:%PORT%
exit /b 0
