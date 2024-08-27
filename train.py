from ultralytics import YOLO

def train_model():
    # Initialize the YOLO model
    model = YOLO('yolov8m.pt')  # Load the pre-trained model

    # Define the training configuration
    data = 'data_custom.yaml'  # Path to your custom dataset YAML file
    epochs = 1  # Number of training epochs
    imgsz = 640  # Image size
    batch = 8  # Batch size

    # Train the model
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        task='detect',
        mode='train'
    )

    # Optional: Save the trained model
    model.save('trained_model.pt')

if __name__ == '__main__':
    train_model()
