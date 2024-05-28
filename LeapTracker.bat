@echo off
setlocal

:choose
:: Prompt the user to choose the tracker
echo Choose the tracker to execute:
echo 0 - LeapTracker
echo 1 - WebcamTracker
set /p choice="Enter your choice (0 or 1): "

:: Check the user's choice and run the appropriate Python script
if "%choice%"=="0" (
    goto run_leap
) else if "%choice%"=="1" (
    goto run_webcam
) else (
    echo Invalid choice. Please enter 0 or 1.
    goto choose
)

:run_leap
:: Activate the virtual environment
call venv\Scripts\activate
:: Run LeapTracker.py
python LeapTracker.py
:: Deactivate the virtual environment
deactivate
goto end

:run_webcam
:: Activate the virtual environment
call venv\Scripts\activate
:: Run WebcamTracker.py
python WebcamTracker.py
:: Deactivate the virtual environment
deactivate
goto end

:end
endlocal
