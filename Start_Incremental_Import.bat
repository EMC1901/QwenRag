@echo off
rem Customer-facing entry point. Double-click this file.
call "%~dp0scripts\submit_incremental_import.bat"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%
