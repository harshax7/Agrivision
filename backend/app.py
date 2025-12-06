from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# ---------------------------
# Load YOLO Leaf Detection
# ---------------------------
print("Loading YOLO model...")
yolo_model = YOLO('best_tomato_leaf_model.pt')
print("YOLO loaded!")

# ---------------------------
# Load TFLite Disease Model
# ---------------------------
print("Loading TFLite disease model...")

interpreter = tf.lite.Interpreter(model_path="disease_detection_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully!")

# Disease class list
disease_classes = [
    'Tomato_Bacterial_spot', 
            'Tomato_Early_blight',
            'Tomato_Late_blight',
            'Tomato_Leaf_Mold',
            'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites',
            'Tomato_Target_Spot',
            'Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato_mosaic_virus',
            'Tomato_healthy'
]

# ----------------------------------------------------
# Helper: TFLite inference
# ----------------------------------------------------
# def run_tflite(image_np):
#     img = cv2.resize(image_np, (224, 224))
#     img = img.astype(np.float32) / 255.0
#     img = np.expand_dims(img, axis=0)

#     interpreter.set_tensor(input_details[0]['index'], img)
#     interpreter.invoke()
#     output = interpreter.get_tensor(output_details[0]['index'])[0]

#     max_index = np.argmax(output)
#     return {
#         "disease": disease_classes[max_index],
#         "confidence": round(float(output[max_index]) * 100, 2),
#         "raw": output.tolist()
#     }

def run_tflite(image_np):
    img = cv2.resize(image_np, (224, 224))
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0–1 float conversion
    img = tf.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img.numpy())
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    max_index = np.argmax(output)

    return {
        "disease": disease_classes[max_index],
        "confidence": round(float(output[max_index]) * 100, 2),
        "raw": output.tolist()
    }

# ----------------------------------------------------
# API: Disease Detection
# ----------------------------------------------------
@app.route('/detect-disease', methods=['POST'])
def detect_disease():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        file = request.files['image']
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)

        # ----------------------
        # Step 1: YOLO leaf detection
        # ----------------------
        results = yolo_model(image_np, conf=0.7)
        leaf_detected = len(results[0].boxes) > 0

        if not leaf_detected:
            return jsonify({
                "success": True,
                "leaf_detected": False,
                "message": "No tomato leaf detected"
            })

        # ----------------------
        # Step 2: Disease TFLite detection
        # ----------------------
        disease_result = run_tflite(image_np)

        return jsonify({
            "success": True,
            "leaf_detected": True,
            "disease": disease_result["disease"],
            "confidence": disease_result["confidence"],
            "raw_predictions": disease_result["raw"]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/health')
def health():
    return {
        "status": "running",
        "yolo_loaded": True,
        "tflite_loaded": True
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
