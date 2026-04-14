# moveInARSpace

This project uses MediaPipe and OpenCV to create an augmented reality (AR) space where you can draw in real-time using hand gestures. The application tracks hand landmarks and allows you to draw on the screen by moving your fingers.

## Usage

1. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the ball mover application:

   ```bash
    python main.py
    ```

3. Or run the drawing application:

   ```bash
    python draw.py
    ```

4. Or run the pointer ball application:

   ```bash
    python pointer_ball.py
    ```

5. To exit the application, press the 'q' key.

6. In ```pointer_ball.py```, you need to press and hold the 'c' key to make the ball visible.

7. In ```main.py```, press 'r' to reset the ball's position.

8. In ```draw.py```, press 'c' to clear the drawing.

9. In ```draw.py```, press 'u' to undo the last drawn point.

## Notes

You need to download the MediaPipe Hand Landmarker model and place it in the same directory as the `main.py` file. You can download the model from the following link:
[MediaPipe Hand Landmarker Model](https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task)
