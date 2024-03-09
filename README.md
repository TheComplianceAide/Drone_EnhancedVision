# Drone Vision Enhancement Toolkit

This toolkit comprises several Python scripts designed to enhance and manage the visual capabilities of consumer drones, making it ideal for hobbyists, researchers, and developers. The scripts focus on real-time image processing, motion detection, object tracking, and dynamic image enhancement, optimized for running on consumer hardware like the Microsoft Surface Pro.

## Overview

The collection includes scripts for:
- Real-time motion detection and object tracking.
- Edge detection for improved visual analysis.
- Dynamic image enhancements including brightness, contrast adjustments, and advanced noise reduction.
- A Tkinter-based application launcher, `app_launcher_v2.py`, that allows users to manage and execute the aforementioned scripts dynamically.

Each script is designed to operate independently or in conjunction to provide a comprehensive suite of tools for drone vision applications.

## Setup

1. **Install Python**: Ensure you have Python 3.x installed on your system.

2. **Install Dependencies**: The scripts rely on various third-party libraries such as OpenCV, MSS, Numpy, Psutil, and Pillow. Install them using pip:

    ```
    pip install opencv-python mss numpy psutil Pillow
    ```

3. **Download the Toolkit**: Clone or download the toolkit repository to your local machine.

4. **Place Scripts in a Common Directory**: Ensure all scripts are located in a common directory, such as `C:\temp\Drone_CV_Vision`. The app launcher script, `app_launcher_v2.py`, will dynamically list and allow execution of scripts from this directory.

## Running the App Launcher

The Tkinter app launcher, `app_launcher_v2.py`, provides a graphical interface for launching and managing the drone vision scripts. Follow these steps to use the launcher:

1. **Launch the App Launcher**: Navigate to the directory containing the scripts and run the `app_launcher_v2.py` script:

    ```
    python app_launcher_v2.py
    ```

2. **Select and Launch Scripts**: The launcher will display buttons for each Python script prefixed with an underscore (`_`). Click on the button corresponding to the script you wish to run. The selected script will launch in a new window.

3. **Terminate Running Scripts**: Use the "Kill script" button within the app launcher to terminate the currently running script before starting another.

## Integrating Scripts with Consumer Drones

To integrate these scripts with consumer drones, follow these guidelines:

1. **Capture Feed**: Utilize the drone's video output as an input for the image processing scripts. This may require custom configuration depending on the drone's model and the method of video feed access (e.g., RTSP stream).

2. **Enhance and Analyze**: Select the desired image enhancement or analysis script from the app launcher. For example, use motion detection for surveillance applications or dynamic range compression for improving visibility in varied lighting conditions.

3. **Optimize for Hardware**: The scripts are optimized for consumer hardware, like the Surface Pro. Adjust parameters within each script (e.g., resolution settings in motion detection) to match your hardware's capabilities for optimal performance.

## Customizing the Toolkit

Feel free to customize the app launcher and individual scripts to suit your specific requirements. The modular design allows for easy expansion, adjustment of parameters, and incorporation of new features to enhance your drone's vision capabilities further.

## Contribution

Contributions to this toolkit are welcome! Whether it's adding new features, improving existing scripts, or reporting issues, your input helps make this toolkit more valuable to everyone.
