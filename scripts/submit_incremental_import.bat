@echo off
setlocal
rem Customer entry point: no Conda activation or command-line knowledge required.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0submit_incremental_import.ps1" -WaitForCompletion
exit /b %ERRORLEVEL%
