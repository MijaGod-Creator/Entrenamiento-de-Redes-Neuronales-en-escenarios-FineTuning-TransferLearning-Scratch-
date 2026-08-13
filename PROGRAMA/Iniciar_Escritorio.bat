@echo off
title APLICACION DE ESCRITORIO - RECONOCIMIENTO DE EMOCIONES
color 0B
echo =====================================================================
echo    INICIANDO INTERFAZ DE ESCRITORIO (GUI) DEL PROYECTO FER
echo =====================================================================
echo.

:: Verificar si el entorno virtual existe
if not exist .venv\Scripts\activate.bat (
    color 0C
    echo ERROR: No se ha detectado la instalacion previa.
    echo Por favor, ejecuta primero "Instalar_Programa.bat" para instalar el programa.
    echo.
    pause
    exit /b
)

echo [+] Activando entorno virtual...
call .venv\Scripts\activate.bat

echo [+] Iniciando aplicacion de escritorio nativa...
python gui_app.py

echo.
echo [+] Aplicacion cerrada de forma segura.
choice /d y /t 1 >nul
