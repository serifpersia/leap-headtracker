import numpy as np        # `pip install numpy`
import cv2                # `pip install opencv-python`
import pyvirtualcam      # `pip install pyvirtualcam`
import leapuvc            # Ensure leapuvc.py is in this folder

# Define a function to apply zoom
def apply_zoom(image, zoom_factor):
    if zoom_factor == 1.0:
        return image
    height, width = image.shape[:2]
    new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)
    y1 = (height - new_height) // 2
    x1 = (width - new_width) // 2
    cropped_image = image[y1:y1+new_height, x1:x1+new_width]
    resized_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image

# Retrieve calibration data using DSHOW backend
capResolution = (640, 480)
cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, capResolution[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, capResolution[1])
calibration = leapuvc.retrieveLeapCalibration(cam, capResolution)
cam.release()

# Start the Leap Capture Thread with MSMF backend
leap = leapuvc.leapImageThread(resolution=(640, 480))
leap.start()
leap.calibration = calibration

cam_id = 0
rectify_on = 0

# Create a virtual camera with the specified backend
with pyvirtualcam.Camera(width=640, height=480, fps=60) as cam:
    print(f'Using virtual camera: {cam.device}')
    
    # Define Various Camera Control Settings
    cv2.namedWindow('Settings')
    cv2.resizeWindow('Settings', 400, 385)
    cv2.moveWindow('Settings', 0, 0)
    cv2.createTrackbar('Camera', 'Settings', cam_id, 1, lambda x: None)
    cv2.createTrackbar('Rectify', 'Settings', rectify_on, 1, lambda x: None)
    cv2.createTrackbar('Exposure',  'Settings', 1000,  32222, leap.setExposure)   # Sets the exposure time in microseconds
    cv2.createTrackbar('LEDs',      'Settings', 0, 1,  lambda a: (leap.setLeftLED(a), leap.setCenterLED(a), leap.setRightLED(a))) # Turns on the IR LEDs
    cv2.createTrackbar('Gamma',     'Settings', 1, 1,  leap.setGammaEnabled)      # Applies a sqrt(x) contrast-reducing curve in 10-bit space
    cv2.createTrackbar('Anlg Gain', 'Settings', 0, 63, leap.setGain)              # Amplifies the signal in analog space, 16-63
    cv2.createTrackbar('Dgtl Gain', 'Settings', 0, 16, leap.setDigitalGain)       # Digitally amplifies the signal in 10-bit space
    cv2.createTrackbar('HDR',       'Settings', 0, 1,  leap.setHDR)               # Selectively reduces the exposure of bright areas at the cost of fixed-pattern noise
    cv2.createTrackbar('Rotation',  'Settings', 0, 1,  leap.set180Rotation)       # Rotate the image 180 degrees
    cv2.createTrackbar('Zoom', 'Settings', 2, 10, lambda x: None)  # Zoom level

    # Capture images until the program is manually closed
    while leap.running:
        newFrame, left_right_image = leap.read()
        if newFrame:
            # Get the current zoom level from the trackbar
            zoom_level = cv2.getTrackbarPos('Zoom', 'Settings') + 1.0
            rectify_on = cv2.getTrackbarPos('Rectify', 'Settings')
            cam_id = cv2.getTrackbarPos('Camera', 'Settings')

            CAM_INDEX = 0
            
            if cam_id:
                CAM_INDEX = 0
            else:
                CAM_INDEX = 1
                
            if rectify_on:
                if CAM_INDEX == 1:
                    calibration = 'right'
                else:
                    calibration = 'left'
                maps = leap.calibration[calibration]["undistortMaps"]
                rectified_image = cv2.remap(left_right_image[CAM_INDEX], maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
                processed_image = rectified_image
            else:
                processed_image = left_right_image[CAM_INDEX]
                
            # Apply zoom to the left camera image
            zoomed_image = apply_zoom(processed_image, zoom_level)
            zoomed_image = cv2.merge((zoomed_image, zoomed_image, zoomed_image))

            # Send the frame to the virtual camera
            cam.send(zoomed_image)
        if cv2.waitKey(1) == ord('q'):  # Check if the 'Esc' key is pressed (27 is the ASCII code for 'Esc')
            break

    cv2.destroyAllWindows()
