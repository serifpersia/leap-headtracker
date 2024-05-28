<div align="center">

![leaptracker](https://github.com/serifpersia/leap-headtracker/assets/62844718/515eddf8-9f61-4832-a2f9-3827e5781023)

<h1><span class="piano-text" style="color: white;">Leap Tracker</span>

[![Release](https://img.shields.io/github/release/serifpersia/leap-headtracker.svg?style=flat-square)](https://github.com/serifpersia/leap-headtracker-esp32/releases)
[![License](https://img.shields.io/github/license/serifpersia/leap-headtracker?color=blue&style=flat-square)](https://raw.githubusercontent.com/serifpersia/leap-headtracker/master/LICENSE)

</div>

Leap Tracker is a head tracking application that uses Leap Motion controller as the video source. It enables users to track head movements in real-time and transmit the tracking data over UDP network. The tracked data can be used with opentrack software.

## Download
Download latest release [here](https://github.com/serifpersia/leap-headtracker/releases).

## Requirements
Leap Motion Controller needs to be unlocked to be used as a UVC device.
Unlock and Restore tools are provided and Windows machine is needed to use executables to unlock or restore Leap Motion Controller device.
To use Leap Tracker you need a newer version of Python installed `3.10, 3.12`.

## Installation
- Download the release ZIP file or clone the repository to your local machine.
- Navigate to the extracted zip directory.
- Run the install.bat on Windows or install.sh on Linux. This will set up the necessary environment and install dependencies.

## Usage
- Connect Leap Motion Controller device to available USB Port.
- Run Leap Tracker application using `LeapTracker.bat` on Windows or `LeapTracker.sh` on Linux
- Leap Motion Controller video feed preview should be seen, use Settings button to adjust the image feed for better tracking `exposure, gamma, leds, rotatiom, zoom`.
Launch opentrack software and use UDP over network as the source for tracking data, press `Start` button and the pink octopus tracking indicator should now move as well as pitch, yaw and roll data points should change based on head movement. Adjust Mapping curves for pitch yaw and roll, for most stable tracking only enable yaw tracking and adjust the curve to desired degrees for your game.Keybinding shortcut for resetting tracking position is recommended in opentrack.
- You can use `Q` keybind when Leap Tracker window is focused window to quit the app or hold `R` keybind for few seconds to recalibrate center pose.

## License
This project is licensed under the [MIT License](LICENSE).
