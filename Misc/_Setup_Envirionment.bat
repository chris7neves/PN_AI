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
::CHECK IF 3.6.0 x64 IS INSTALLED
call :colorEcho b0 "CHECK PYTHON 3.6.0 x64"
echo.
echo F | xcopy /y /e /s /c %HOME_DIRECTORY%requirements.txt %PYTHON36X64%\Scripts\requirements.txt
IF NOT EXIST %PYTHON36X64% GOTO NOPYTHON36X64
cd %PYTHON36X64%

::Update PIP Install
IF EXIST python.exe (
call :colorEcho 0a "PYTHON.EXE FOUND"
echo.
python.exe -m pip install --upgrade pip
) ELSE (
call :colorEcho 0c "PYTHON.EXE NOT FOUND - UNABLE TO UPDATE PIP"
echo.
)

cd %PYTHON36X64%\Scripts
IF EXIST pip3.exe (
call :colorEcho 0a "PIP3.EXE FOUND"
echo.
::LIST PACKAGES TO INSTALL HERE IN REQUIREMENTS.TXT
pip3.exe install -r requirements.txt

call :colorEcho a0 "SUCCESSFULLY UPDATED"
echo.

) ELSE (
call :colorEcho 0c "PIP3.EXE MISSING - CHECK PATHS - PYTHON VERSION NOT UPDATED"
echo.
GOTO NOPYTHON36X64
)
:NOPYTHON36X64
::============================================================================

echo.
echo.

::============================================================================
::CHECK IF 3.7.0 x32 IS INSTALLED
call :colorEcho b0 "CHECK PYTHON 3.7.0 x32"
echo.
echo F | xcopy /y /e /s /c %HOME_DIRECTORY%requirements.txt %PYTHON37X32%\Scripts\requirements.txt
IF NOT EXIST %PYTHON37X32% GOTO NOPYTHON37X32
cd %PYTHON37X32%

::Update PIP Install
IF EXIST python.exe (
call :colorEcho 0a "PYTHON.EXE FOUND"
echo.
python.exe -m pip install --upgrade pip
) ELSE (
call :colorEcho 0c "PYTHON.EXE NOT FOUND - UNABLE TO UPDATE PIP"
echo.
)

cd %PYTHON37X32%\Scripts
IF EXIST pip3.exe (
call :colorEcho 0a "PIP3.EXE FOUND"
echo.
::LIST PACKAGES TO INSTALL HERE
pip3.exe install -r requirements.txt


call :colorEcho a0 "SUCCESSFULLY UPDATED"
echo.

) ELSE (
call :colorEcho 0c "PIP3.EXE MISSING - CHECK PATHS - PYTHON VERSION NOT UPDATED"
echo.
GOTO NOPYTHON37X32
)
:NOPYTHON37X32
::============================================================================

echo.
echo.

::============================================================================
::CHECK IF 3.7.0 x64 IS INSTALLED
call :colorEcho b0 "CHECK PYTHON 3.7.0 x64"
echo.
echo F | xcopy /y /e /s /c %HOME_DIRECTORY%requirements.txt %PYTHON37X64%\Scripts\requirements.txt
IF NOT EXIST %PYTHON37X64% GOTO NOPYTHON37X64
cd %PYTHON37X64%

::Update PIP Install
IF EXIST python.exe (
call :colorEcho 0a "PYTHON.EXE FOUND"
echo.
python.exe -m pip install --upgrade pip
) ELSE (
call :colorEcho 0c "PYTHON.EXE NOT FOUND - UNABLE TO UPDATE PIP"
echo.
)

cd %PYTHON37X64%\Scripts
IF EXIST pip3.exe (
call :colorEcho 0a "PIP3.EXE FOUND"
echo.
::LIST PACKAGES TO INSTALL HERE
pip3.exe install -r requirements.txt


call :colorEcho a0 "SUCCESSFULLY UPDATED"
echo.

) ELSE (
call :colorEcho 0c "PIP3.EXE MISSING - CHECK PATHS - PYTHON VERSION NOT UPDATED"
echo.
GOTO NOPYTHON37X64
)
:NOPYTHON37X64
::============================================================================

echo.
echo.

::============================================================================
::CHECK IF 3.8.0 x64 IS INSTALLED
call :colorEcho b0 "CHECK PYTHON 3.8.0 x64"
echo.
echo F | xcopy /y /e /s /c %HOME_DIRECTORY%requirements.txt %PYTHON38X64%\Scripts\requirements.txt
IF NOT EXIST %PYTHON38X64% GOTO NOPYTHON38X64
cd %PYTHON38X64%

::Update PIP Install
IF EXIST python.exe (
call :colorEcho 0a "PYTHON.EXE FOUND"
echo.
python.exe -m pip install --upgrade pip
) ELSE (
call :colorEcho 0c "PYTHON.EXE NOT FOUND - UNABLE TO UPDATE PIP"
echo.
)

cd %PYTHON37X32%\Scripts
IF EXIST pip3.exe (
call :colorEcho 0a "PIP3.EXE FOUND"
echo.
::LIST PACKAGES TO INSTALL HERE
pip3.exe install -r requirements.txt


call :colorEcho a0 "SUCCESSFULLY UPDATED"
echo.

) ELSE (
call :colorEcho 0c "PIP3.EXE MISSING - CHECK PATHS - PYTHON VERSION NOT UPDATED"
echo.
GOTO NOPYTHON38X64
)
:NOPYTHON38X64
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
