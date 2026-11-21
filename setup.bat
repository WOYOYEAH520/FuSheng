@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

rem ---- 找一个 Python 3.12 ----
set "BASEPY=py -3.12"
%BASEPY% -V >nul 2>&1
if errorlevel 1 set "BASEPY=python"
%BASEPY% -V >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 没有找到 Python，请先安装 Python 3.12：
    echo         https://www.python.org/downloads/
    echo         安装时记得勾选 "Add python.exe to PATH"。
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] 创建虚拟环境 .venv ...
    %BASEPY% -m venv .venv
    if errorlevel 1 goto failed
)

echo [2/5] 升级 pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed

echo [3/5] 安装 PyTorch 2.4.0 + CUDA 12.1（约 2.5 GB，请耐心等）...
".venv\Scripts\python.exe" -m pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto failed

echo [4/5] 安装其余依赖 ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo [5/5] 安装 BS-RoFormer 后端（pymss）...
rem 加 --no-deps：pymss 的元数据要求 torch>=2.7.1，装它的依赖会把 torch 升级掉
".venv\Scripts\python.exe" -m pip install --no-deps pymss==2.1.12 pymss-core==0.1.14
if errorlevel 1 goto failed

echo.
echo 安装完成，双击 启动.bat 即可运行。
pause
exit /b 0

:failed
echo.
echo [ERROR] 安装失败，检查网络后重跑本文件，或把上面的报错发出来。
pause
exit /b 1
