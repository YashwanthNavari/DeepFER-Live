import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os
from collections import deque

def load_emotion_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return None
    return tf.keras.models.load_model(model_path)

def draw_emotion_bars(frame, emotion_labels, probs):
    """Draws a beautiful bar chart overlay of emotion probabilities on the frame."""
    bar_w = 150
    bar_h = 12
    start_x = 10
    start_y = 60
    
    # Draw background panel
    cv2.rectangle(frame, (start_x - 5, start_y - 20), (start_x + bar_w + 110, start_y + (len(emotion_labels) * 25) + 5), (0, 0, 0), -1)
    # Give panel some transparency by blending
    
    for i, (label, prob) in enumerate(zip(emotion_labels, probs)):
        text_y = start_y + (i * 25)
        # Label
        cv2.putText(frame, label.capitalize(), (start_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Background bar
        cv2.rectangle(frame, (start_x + 65, text_y - 10), (start_x + 65 + bar_w, text_y + 5), (50, 50, 50), -1)
        
        # Color depends on emotion
        color = (0, 255, 0) # Green default
        if label == 'happy': color = (0, 255, 255) # Yellow
        elif label == 'angry': color = (0, 0, 255) # Red
        elif label == 'sad': color = (255, 0, 0) # Blue
        elif label == 'surprise': color = (255, 255, 0) # Cyan
        
        # Filled bar
        fill_w = int(bar_w * prob)
        if fill_w > 0:
            cv2.rectangle(frame, (start_x + 65, text_y - 10), (start_x + 65 + fill_w, text_y + 5), color, -1)
            
        # Percentage text
        cv2.putText(frame, f"{prob*100:.1f}%", (start_x + 65 + bar_w + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def main():
    # Classes based on FER2013 standard
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_model.keras')
    model = load_emotion_model(model_path)
    if model is None:
        return

    # MediaPipe Face Detection
    has_mp = False
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        has_mp = True
    except Exception as e:
        print(f"MediaPipe Face Detection not available. Error: {e}")
        print("Using OpenCV fallback.")

    # Fallback OpenCV Haar Cascade
    cascade_local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_local_path):
        import urllib.request
        cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(cascade_url, cascade_local_path)
        except Exception as e:
            pass
            
    face_cascade = cv2.CascadeClassifier(cascade_local_path)

    # Use DirectShow backend which is much more stable on Windows for unlocking cameras
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam. Press 'q' to exit.")

    prev_time = 0
    # Queue to smooth predictions over time (prevents flickering)
    emotion_history = deque(maxlen=5) 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        faces_detected = False
        final_prediction = None
        box_coords = None

        if has_mp:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results and results.detections:
                faces_detected = True
                # Process the first prominent face
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                fw = int(bboxC.width * w)
                fh = int(bboxC.height * h)
                
                x, y = max(0, x), max(0, y)
                fw, fh = min(w - x, fw), min(h - y, fh)
                
                if fw > 0 and fh > 0:
                    roi_gray = gray_frame[y:y+fh, x:x+fw]
                    box_coords = (x, y, fw, fh)

        # Fallback to Haar Cascade if MediaPipe fails
        if not faces_detected and not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
            if len(faces) > 0:
                (x, y, fw, fh) = faces[0]
                # Tight crop for Haar Cascade
                y_tight = y + int(fh * 0.1)
                fh_tight = int(fh * 0.8)
                x_tight = x + int(fw * 0.1)
                fw_tight = int(fw * 0.8)
                
                roi_gray = gray_frame[y_tight:y_tight+fh_tight, x_tight:x_tight+fw_tight]
                box_coords = (x_tight, y_tight, fw_tight, fh_tight)
                faces_detected = True

        # If a face was found, run prediction
        if faces_detected and box_coords is not None:
            (bx, by, bfw, bfh) = box_coords
            if roi_gray.size > 0:
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = cv2.equalizeHist(roi_gray)
                roi_normalized = roi_gray / 255.0
                roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

                prediction = model.predict(roi_reshaped, verbose=0)[0]
                emotion_history.append(prediction)
                
                # Smooth predictions
                avg_prediction = np.mean(emotion_history, axis=0)
                max_index = int(np.argmax(avg_prediction))
                confidence = avg_prediction[max_index]
                predicted_emotion = emotion_labels[max_index]

                # Draw UI
                cv2.rectangle(frame, (bx, by), (bx + bfw, by + bfh), (0, 255, 0), 2)
                text = f"{predicted_emotion.capitalize()} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                draw_emotion_bars(frame, emotion_labels, avg_prediction)
        else:
            # Clear history if no face seen
            emotion_history.clear()

        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('DeepFER Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
