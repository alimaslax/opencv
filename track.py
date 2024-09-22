from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "./input.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Initialize frame counter
frame_count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count += 1
        
        # Process every 4th frame
        if frame_count % 4 == 0:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.predict(source=frame, conf=0.25)

            # Get the boxes and class IDs
            boxes = results[0].boxes.xywh.cpu()
            class_ids = results[0].boxes.cls.cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for i, box in enumerate(boxes):
                x, y, w, h = box
                class_id = class_ids[i]
                track = track_history[class_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()