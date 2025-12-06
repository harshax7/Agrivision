import os
from ultralytics import YOLO

def train_yolo():
    DATASET_PATH = r"C:\Users\harsh\OneDrive\Desktop\Agrolens_major\YOlov8"
    yaml_path = f"{DATASET_PATH}/data.yaml"

    class_names = [
        "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
        "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
        "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus",
        "Tomato Healthy", "Tomato Mosaic Virus"
    ]

    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    print("Starting training...")

    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device="cuda",  # Use GPU
        workers=0        # 🔥 REQUIRED FOR WINDOWS
    )

    print("Saving model...")
    model.save(r"C:\Users\harsh\OneDrive\Desktop\Agrolens_major\best_tomato_leaf_model.pt")

    print("Evaluating model...")
    metrics = model.val(data=yaml_path)

    print("\n Model Evaluation Metrics:")
    print(f" Precision: {metrics.box.mp:.3f}")
    print(f" Recall: {metrics.box.mr:.3f}")
    print(f" mAP@50: {metrics.box.map50:.3f}")
    print(f" mAP@50-95: {metrics.box.map:.3f}")

if __name__ == '__main__':
    train_yolo()



