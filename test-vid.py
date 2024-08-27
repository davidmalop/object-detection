from ultralytics import YOLO
import cv2

model = YOLO("trained_model.pt")

conf_threshold = 0.5

source = "vid2.mp4"
cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=conf_threshold)

    value = results[0]
    num_pieces = len(results[0].boxes) 
    print(num_pieces)

    text = f"Pieces detected: {num_pieces}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    annotated_frame = results[0].plot()  

    cv2.imshow("Object Detection", annotated_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
