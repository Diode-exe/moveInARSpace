import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global variable to store results
latest_result = None
draw_list = []  # List to store points for drawing

def update_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Initialize Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=update_result
)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Mirror the frame for a more natural "AR" feel
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Send to AI
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        # MANUAL DRAWING LOGIC
        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                for landmark in hand_landmarks:
                    # Convert normalized (0.0 to 1.0) to pixel coordinates
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    # Draw a circle on the joint
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # 1. Compare the Y-coordinates of the tip (8) and the knuckle (6)
                # Note: In MediaPipe, Y decreases as you move UP the screen.
                if hand_landmarks[8].y < hand_landmarks[6].y:

                    # 2. Convert normalized coordinates to pixel values
                    x = int(hand_landmarks[8].x * frame.shape[1])
                    y = int(hand_landmarks[8].y * frame.shape[0])

                    # 3. Add to your list
                    draw_list.append((x, y))
                    print("finger up")

                else:
                    print("finger down")

                # if hand_landmarks[20].y < hand_landmarks[18].y:
                #     x = int(hand_landmarks[20].x * frame.shape[1])
                #     y = int(hand_landmarks[20].y * frame.shape[0])
                #     if (x, y) in draw_list:
                #         draw_list.remove((x, y))

        # Draw lines between points in draw_list
        for i in range(1, len(draw_list)):
            cv2.line(frame, draw_list[i-1], draw_list[i], (255, 0, 0), 2)

        cv2.imshow('AR Space - Manual Drawing', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
