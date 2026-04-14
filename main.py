import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MoveBallHand:
    """Class to handle hand landmark detection and drawing on the video feed."""
    def __init__(self):
        self.hand_model = "hand_landmarker.task"
        self.latest_result = None
        self.base_options = None
        self.options = None
        self.base_options = python.BaseOptions(model_asset_path=self.hand_model)
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
                landmarker.detect_async(mp_image, int(time.time() * 1000))
                if self.latest_result and self.latest_result.hand_landmarks:
                    for hand_landmarks in self.latest_result.hand_landmarks:
                        for landmark in hand_landmarks:
                            # Convert normalized (0.0 to 1.0) to pixel coordinates
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])

                            # Draw a circle on the joint
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                            if cv2.waitKey(1) & 0xFF == ord('c'):
                                cv2.circle(frame, (100, y), 20, (255, 0, 0), -1)
                cv2.imshow('Move Ball', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

ballmove = MoveBallHand()
ballmove.start_drawing()
