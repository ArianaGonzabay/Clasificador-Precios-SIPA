@echo off
cd /d "%~dp0"
taskkill /f /im streamlit.exe >nul 2>&1
"%~dp0.venv\Scripts\streamlit.exe" run app.py