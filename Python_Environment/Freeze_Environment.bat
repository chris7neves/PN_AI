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
::===========================================================================


::============================================================================
call :colorEcho b0 "FREEZING PYTHON ENVIRONMENT"
echo.

::LIST PACKAGES TO INSTALL HERE
pip3.exe freeze > requirements.txt

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
