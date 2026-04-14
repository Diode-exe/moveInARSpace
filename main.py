import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDrawing:
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
        with vision.HandLandmarker.create_from_options(self.options) as landmarker:
            cap = cv2.VideoCapture(0)

            while cap.isOpened():
                success, frame = cap.read()
                if not success: break

                # Mirror the frame for a more natural "AR" feel
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                landmarker.detect_async(mp_image, int(time.time() * 1000))
                cv2.imshow('AR Space - Manual Drawing', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

hand_drawer = HandDrawing()
hand_drawer.start_drawing()
