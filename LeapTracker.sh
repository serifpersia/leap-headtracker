#!/bin/bash

while true; do
    # Prompt the user to choose the tracker
    echo "Choose the tracker to execute:"
    echo "0 - LeapTracker"
    echo "1 - WebcamTracker"
    read -p "Enter your choice (0 or 1): " choice

    if [ "$choice" == "0" ]; then
        # Activate the virtual environment
        source venv/bin/activate
        # Run LeapTracker.py
        python LeapTracker.py
        # Deactivate the virtual environment
        deactivate
        break
    elif [ "$choice" == "1" ]; then
        # Activate the virtual environment
        source venv/bin/activate
        # Run WebcamTracker.py
        python WebcamTracker.py
        # Deactivate the virtual environment
        deactivate
        break
    else
        echo "Invalid choice. Please enter 0 or 1."
    fi
done
