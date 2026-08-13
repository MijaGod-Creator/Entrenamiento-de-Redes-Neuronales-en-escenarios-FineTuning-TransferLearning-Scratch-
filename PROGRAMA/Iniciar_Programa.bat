@echo off
title INICIAR RECONOCIMIENTO DE EMOCIONES
color 0B
echo =====================================================================
echo    INICIANDO EL SISTEMA DE RECONOCIMIENTO DE EMOCIONES (FER)
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

echo [+] Levantando el servidor local en segundo plano...
:: Lanzar Flask en segundo plano redirigiendo a logs
start /b python ..\app.py > server.log 2>&1

echo [+] Esperando 3 segundos a que el modelo cargue en memoria...
choice /d y /t 3 >nul

echo [+] Abriendo el navegador en la aplicacion local...
start http://localhost:5000

echo.
color 0A
echo =====================================================================
echo    ¡SERVIDOR EJECUTANDOSE CON EXITO EN http://localhost:5000!
echo =====================================================================
echo.
echo Presiona cualquier tecla en esta ventana para APAGAR el servidor
echo y cerrar el programa.
echo.
pause

:: Cerrar procesos de python locales del venv
echo [+] Apagando el servidor local...
taskkill /f /im python.exe >nul 2>&1
echo [+] Servidor apagado. ¡Hasta pronto!
choice /d y /t 1 >nul
