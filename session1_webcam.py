import cv2
import time

# 1. face
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 2. open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Can not open camera!")
    exit()

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. information
    height, width = frame.shape[:2]

    # ========== Video processing ==========
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 4. face detection_1
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # 5. face frame
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )

    # ========== FPS ==========
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
    prev_time = curr_time

    # ========== information ==========
    info_text = [
        f"Resolution: {width}x{height}",
        f"FPS: {fps}",
        f"Faces detected: {len(faces)}",
        "Press Q to quit"
    ]

    y0 = 25
    for i, text in enumerate(info_text):
        cv2.putText(
            frame,
            text,
            (10, y0 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # ========== second page ==========
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # video overlap
    overlay = cv2.addWeighted(gray_colored, 0.7, edges_colored, 0.3, 0)

    # ========== two page ==========
    cv2.imshow("Camera - Face Detection", frame)
    cv2.imshow("Gray + Edge Overlay", overlay)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释release
cap.release()
cv2.destroyAllWindows()
