import sys
import cv2
import json
import time

# Default to camera index 1 (your Insta360), or use the index passed in terminal (e.g. python asl_stream.py 0)
cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print(f"Error: Could not open camera stream at index {cam_index}.")
    print("Make sure the Insta360 is in Webcam Mode or try another index (e.g., python3 asl_stream.py 0)")
    exit()

print(f"Starting Insta360 Stream on camera index {cam_index}... Press 'q' in the preview window to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Flip horizontally for natural mirror preview
    frame = cv2.flip(frame, 1)

    # Display connection status text on video stream
    cv2.putText(frame, "Insta360 -> Claude Code Active", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save current frame snapshot and json metrics for Claude Code
    cv2.imwrite("latest_frame.jpg", frame)
    stream_payload = {
        "timestamp": time.time(),
        "status": "connected",
        "camera_active": True,
        "camera_index": cam_index
    }
    with open("asl_output.json", "w") as f:
        json.dump(stream_payload, f, indent=2)

    # Show live camera popup window
    cv2.imshow("Insta360 ASL Stream", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()