:: Setup Python Environment script
:: Description: This script should be run to install all required packages for python
::
::
::
:: Current Python Versions: 3.7.0, 3.
::
::
@echo off
SETLOCAL EnableDelayedExpansion
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do     rem"') do (
  set "DEL=%%a"
)
color E

call :colorEcho 0f "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo.
call :colorEcho 0f "STARTING PYTHON UPDATER SCRIPT"
echo.
call :colorEcho 0f "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo.
echo.
echo.

::===========================================================================
:: SET PATHS HERE
SET HOME_DIRECTORY=%~dp0

::PYTHON VERSIONS
SET PYTHON35X32=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python35-32
SET PYTHON36X64=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python36
SET PYTHON37X64=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python37
SET PYTHON37X32=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python37-32
SET PYTHON38X64=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python38
::===========================================================================


::============================================================================
call :colorEcho b0 "FREEZING PYTHON ENVIRONMENT VERSION 37x64"
echo.
cd %PYTHON37X64%\Scripts
::COPY PIP FILE TO PYTHON LOCATION
echo F | xcopy /y /e /s /c %HOME_DIRECTORY%Pipfile %PYTHON36X64%\Scripts\Pipfile

pipenv --python %PYTHON37X64%\python.exe
call :colorEcho a0 "REQUIREMENTS FILE CREATED"
echo.

::============================================================================

echo.
echo.

call :colorEcho 0f "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo.
call :colorEcho 0f "SCRIPT COMPLETED"
echo.
call :colorEcho 0f "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo.

pause
EXIT

:colorEcho
echo off
<nul set /p ".=%DEL%" > "%~2"
findstr /v /a:%1 /R "^$" "%~2" nul
del "%~2" > nul 2>&1i
