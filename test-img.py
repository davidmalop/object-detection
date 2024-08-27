import cv2
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO("pieces_detect.pt")

# Function to process an image
def process_image(image):
    # Get predictions from the model
    results = model.predict(image)

    # Count the number of detected objects
    num_objects = len(results)

    # Optionally, filter by class if needed
    # num_objects = sum(1 for result in results if result['class'] == 'desired_class')

    # Display the count on the image
    cv2.putText(image, f"Count: {num_objects}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return image

# Process an example image
image = cv2.imread("2.jpeg")
processed_image = process_image(image)

# Show the result
cv2.imshow("Processed Image", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# from ultralytics import YOLO
# import cv2

# # Load your YOLOv8 model
# model = YOLO("pieces_detect.pt")

# # Set confidence threshold
# conf_threshold = 0.5

# # Load the image
# source = "2.jpeg"
# image = cv2.imread(source)

# # Make predictions on the image
# results = model.predict(image, conf=conf_threshold)

# # Display the results on the image
# annotated_image = results[0].plot()  # Annotate the image with bounding boxes and labels

# # Show the annotated image
# cv2.imshow("YOLOv8 Detection", annotated_image)
# cv2.waitKey(0)  # Wait for a key press to close the image window
# cv2.destroyAllWindows()
