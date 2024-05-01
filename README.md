# Crowd-Detection
This Python script demonstrates real-time crowd detection using YOLOv3 (You Only Look Once) object detection. The system processes video input, detects people within frames, and calculates crowd counts. An alert is triggered if the crowd count exceeds a specified threshold.

This Python script utilizes YOLOv3 (You Only Look Once) for real-time crowd detection in video streams. The system detects people within frames, counts the crowd, and can trigger alerts based on a specified threshold.
**

Key Features**
Object Detection: Detects people in video frames using the YOLOv3 deep learning model.
Crowd Counting: Counts the number of people detected in the scene.
Alerting System: Generates alerts if the crowd count exceeds a predefined threshold.
Modular Design: Organized into modular functions for clarity and ease of modification.

****Usage****

1.Setup:
Ensure Python 3.x is installed.
Install required packages using pip:

----->pip install opencv-python numpy imutils
Download YOLOv3 pre-trained weights (yolov3.weights), configuration (yolov3.cfg), and COCO class labels (coco.names).

2.Run the Script:
Open a terminal or command prompt.
Navigate to the directory containing crowd_detection.py.
Run the script with a video file path:

----->python crowd_detection.py
Adjust the script parameters:
desired_fps: Desired frames per second for video processing.
crowd_threshold: Crowd count threshold for triggering alerts.


****Requirements****
Python 3.x
OpenCV (cv2)
NumPy (numpy)
imutils


----Notes----
Ensure the video file path (video.mp4) is correctly specified in the script.
Experiment with different values of crowd_threshold and desired_fps to optimize performance for specific scenarios.
This script can be extended and integrated into larger applications for crowd monitoring, security systems, or event management.
