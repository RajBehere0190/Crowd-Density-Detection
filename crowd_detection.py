import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

# Load YOLOv3 model and configuration
net = cv2.dnn.readNet("C:\\Users\\91860\\Desktop\\Code\\flask 2023\\yolov3.weights", "C:\\Users\\91860\\Desktop\\Code\\flask 2023\\yolov3.cfg")

# Load COCO class labels
with open("C:\\Users\\91860\\Desktop\\Code\\flask 2023\\coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize video capture (you can replace 0 with your video source)
cap = cv2.VideoCapture('C:\\Users\\91860\\Desktop\\Code\\flask 2023\\video.mp4')

# Set the desired FPS
desired_fps = 30  # Change this value to your desired FPS

# Crowd count threshold
crowd_threshold = 50

while True:
    ret, frame = cap.read()

    # Perform object detection using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Initialize lists to store bounding boxes and confidences
    boxes = []
    confidences = []

    for detection in detections:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.09 and class_id == 0:  # Class ID 0 corresponds to people
            x, y, w, h = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)
            boxes.append([x, y, x + w, y + h])
            confidences.append(float(confidence))

    # Apply non-maximum suppression to eliminate overlapping boxes
    selected_boxes = non_max_suppression(np.array(boxes), np.array(confidences), overlapThresh=0.5)

    # Initialize crowd count
    crowd_count = len(selected_boxes)

    for (startX, startY, endX, endY) in selected_boxes:
        # Draw bounding boxes on the frame
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Check if crowd count exceeds the threshold and generate an alert
    if crowd_count > crowd_threshold:
        alert_message = f"Crowd count exceeded the threshold: {crowd_count}"
        # You can implement alerting logic here, e.g., send an email, SMS, or play a sound

    # Display crowd count on the frame
    cv2.putText(frame, f"Crowd Count: {crowd_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with the desired FPS
    cv2.imshow("Crowd Detection", frame)

    # Adjust the waitKey delay to achieve the desired FPS
    if cv2.waitKey(int(1000 / desired_fps)) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
