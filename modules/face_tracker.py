import cv2
import mediapipe as mp

def detect_face(frame):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
    mp_drawing = mp.solutions.drawing_utils

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    return results

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not accessible")
        exit()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
    mp_drawing = mp.solutions.drawing_utils

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_face(frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )
            cv2.putText(frame, "Face detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face Detection Test', frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        if cv2.getWindowProperty('Face Detection Test', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
