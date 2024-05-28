@echo off

:: Activate the virtual environment
call venv\Scripts\activate

:: Run the Python script
python LeapTracker.py

:: Deactivate the virtual environment
deactivate
