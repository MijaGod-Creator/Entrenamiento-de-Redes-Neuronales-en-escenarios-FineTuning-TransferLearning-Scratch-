@echo off
title INSTALADOR DEL SISTEMA DE RECONOCIMIENTO DE EMOCIONES
color 0B
echo =====================================================================
echo    INSTALADOR DEL SISTEMA DE RECONOCIMIENTO DE EMOCIONES (FER)
echo =====================================================================
echo.
echo Este script creara un entorno virtual de Python e instalara todas
echo las dependencias necesarias de forma automatica y optimizada.
echo.

:: Verificar si Python esta instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo ERROR: Python no esta instalado o no se encuentra en el PATH.
    echo Por favor, instala Python 3.10 o 3.11 desde la Microsoft Store o python.org
    echo e intentalo de nuevo asegurandote de marcar la opcion "Add Python to PATH".
    echo.
    pause
    exit /b
)

echo [+] Creando entorno virtual local (.venv)...
python -m venv .venv
if %errorlevel% neq 0 (
    color 0C
    echo ERROR: No se pudo crear el entorno virtual.
    pause
    exit /b
)

echo [+] Activando entorno virtual...
call .venv\Scripts\activate.bat

echo [+] Actualizando pip y herramientas de instalacion...
python -m pip install --only-binary=:all: --upgrade pip setuptools wheel

echo [+] Instalando stringzilla v3.0 (version compatible sin compilador)...
pip install "stringzilla<3.10"

echo [+] Instalando el resto de dependencias (PyTorch, OpenCV, Flask, Pandas, etc.)...
echo Esto puede demorar unos minutos dependiendo de tu conexion a internet...
pip install -r ..\requirements.txt

echo.
color 0A
echo =====================================================================
echo    ¡INSTALACION COMPLETADA EXITOSAMENTE!
echo =====================================================================
echo.
echo Ahora puedes cerrar esta ventana y ejecutar "Iniciar_Programa.bat"
echo para abrir la aplicacion y probar la deteccion de emociones.
echo.
pause
