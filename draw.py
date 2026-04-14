"""Draw on the live webcam feed with your index finger using MediaPipe's Hand Landmarker.
Press 'c' to clear the drawing and 'q' to quit."""

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class Draw:
    """Class to handle hand landmark detection and drawing on the video feed."""
    def __init__(self):
        self.latest_result = None
        self.base_options = None
        self.draw_list = []  # List to store points for drawing
        self.key = None
        self.base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.update_result
        )

    def update_result(self, result: vision.HandLandmarkerResult,
                      output_image: mp.Image, timestamp_ms: int):
        """Callback function to update the latest hand landmark results."""
        self.latest_result = result

    def start_drawing(self):
        """Start the video capture and drawing loop."""
        with vision.HandLandmarker.create_from_options(self.options) as landmarker:
            cap = cv2.VideoCapture(0)

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Mirror the frame for a more natural "AR" feel
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Send to AI
                timestamp = int(time.time() * 1000)
                landmarker.detect_async(mp_image, timestamp)

                # MANUAL DRAWING LOGIC
                if self.latest_result and self.latest_result.hand_landmarks:
                    for hand_landmarks in self.latest_result.hand_landmarks:
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
                            self.draw_list.append((x, y))
                            print("finger up")

                        else:
                            print("finger down")

                        # if hand_landmarks[20].y < hand_landmarks[18].y:
                        #     x = int(hand_landmarks[20].x * frame.shape[1])
                        #     y = int(hand_landmarks[20].y * frame.shape[0])
                        #     if (x, y) in self.draw_list:
                        #         self.draw_list.remove((x, y))

                # Draw lines between points in draw_list
                for i in range(1, len(self.draw_list)):
                    cv2.line(frame, self.draw_list[i-1], self.draw_list[i], (255, 0, 0), 2)

                cv2.imshow('AR Space - Manual Drawing', frame)

                self.key = cv2.waitKey(1) & 0xFF

                if self.key == ord('q'):
                    break

                if self.key == ord('c'):
                    self.draw_list.clear()  # Clear the drawing when 'c' is pressed
                
                if self.key == ord('u'):
                    if self.draw_list:
                        self.draw_list.pop()  # Remove the last point when 'u' is pressed

            cap.release()
            cv2.destroyAllWindows()

draw = Draw()
draw.start_drawing()