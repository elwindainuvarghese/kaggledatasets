@echo off
title Kaggle Local Clone Server
echo ========================================================
echo Starting Kaggle Local Clone Server...
echo Directory: %~dp0
echo Serving dataset files and full Kaggle UI!
echo ========================================================
python "%~dp0app.py"
pause
