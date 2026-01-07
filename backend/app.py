# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from ultralytics import YOLO
# import numpy as np
# import tensorflow as tf
# import cv2
# from PIL import Image
# import io
# import base64

# app = Flask(__name__)
# CORS(app)

# # ---------------------------
# # Load YOLO Leaf Detection
# # ---------------------------
# print("Loading YOLO model...")
# yolo_model = YOLO('best_tomato_leaf_model.pt')
# print("YOLO loaded!")

# # ---------------------------
# # Load TFLite Disease Model
# # ---------------------------
# print("Loading TFLite disease model...")

# interpreter = tf.lite.Interpreter(model_path="disease_detection_model.tflite")
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# print("TFLite model loaded successfully!")

# # Disease class list
# disease_classes = [
#     'Tomato_Bacterial_spot', 
#             'Tomato_Early_blight',
#             'Tomato_Late_blight',
#             'Tomato_Leaf_Mold',
#             'Tomato_Septoria_leaf_spot',
#             'Tomato_Spider_mites',
#             'Tomato_Target_Spot',
#             'Tomato_Yellow_Leaf_Curl_Virus',
#             'Tomato_mosaic_virus',
#             'Tomato_healthy'
# ]

# # ----------------------------------------------------
# # Helper: TFLite inference
# # ----------------------------------------------------
# # def run_tflite(image_np):
# #     img = cv2.resize(image_np, (224, 224))
# #     img = img.astype(np.float32) / 255.0
# #     img = np.expand_dims(img, axis=0)

# #     interpreter.set_tensor(input_details[0]['index'], img)
# #     interpreter.invoke()
# #     output = interpreter.get_tensor(output_details[0]['index'])[0]

# #     max_index = np.argmax(output)
# #     return {
# #         "disease": disease_classes[max_index],
# #         "confidence": round(float(output[max_index]) * 100, 2),
# #         "raw": output.tolist()
# #     }

# def run_tflite(image_np):
#     img = cv2.resize(image_np, (224, 224))
#     img = tf.image.convert_image_dtype(img, tf.float32)  # 0–1 float conversion
#     img = tf.expand_dims(img, axis=0)

#     interpreter.set_tensor(input_details[0]['index'], img.numpy())
#     interpreter.invoke()

#     output = interpreter.get_tensor(output_details[0]['index'])[0]
#     max_index = np.argmax(output)

#     return {
#         "disease": disease_classes[max_index],
#         "confidence": round(float(output[max_index]) * 100, 2),
#         "raw": output.tolist()
#     }

# # ----------------------------------------------------
# # API: Disease Detection
# # ----------------------------------------------------
# # @app.route('/detect-disease', methods=['POST'])
# # def detect_disease():
# #     try:
# #         if 'image' not in request.files:
# #             return jsonify({"success": False, "error": "No image uploaded"}), 400

# #         file = request.files['image']
# #         image_bytes = file.read()
# #         pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
# #         image_np = np.array(pil_image)

# #         # ----------------------
# #         # Step 1: YOLO leaf detection
# #         # ----------------------
# #         results = yolo_model(image_np, conf=0.7)
# #         leaf_detected = len(results[0].boxes) > 0

# #         if not leaf_detected:
# #             return jsonify({
# #                 "success": True,
# #                 "leaf_detected": False,
# #                 "message": "No tomato leaf detected"
# #             })

# #         # ----------------------
# #         # Step 2: Disease TFLite detection
# #         # ----------------------
# #         disease_result = run_tflite(image_np)

# #         return jsonify({
# #             "success": True,
# #             "leaf_detected": True,
# #             "disease": disease_result["disease"],
# #             "confidence": disease_result["confidence"],
# #             "raw_predictions": disease_result["raw"]
# #         })

# #     except Exception as e:
# #         return jsonify({"success": False, "error": str(e)}), 500

# @app.route('/detect-disease', methods=['POST'])
# def detect_disease():
#     try:
#         # ... (Image loading code stays the same) ...
#         file = request.files['image']
#         image_bytes = file.read()
#         pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         image_np = np.array(pil_image)

#         # ----------------------
#         # Step 1: YOLO leaf detection
#         # ----------------------
#         results = yolo_model(image_np, conf=0.7) # Keep your YOLO confidence high
#         leaf_detected = len(results[0].boxes) > 0

#         if not leaf_detected:
#             return jsonify({
#                 "success": True,
#                 "leaf_detected": False,
#                 "message": "No leaf detected"
#             })
            
#         # ============================================
#         # NEW FIX: Shape/Solidity Check
#         # ============================================
#         if not is_likely_tomato_structure(image_np):
#             return jsonify({
#                 "success": True,
#                 "leaf_detected": False,
#                 "message": "Leaf detected, but shape resembles Hibiscus/Rose (Solid Leaf) rather than Tomato (Compound Leaf)."
#             })
#         # ============================================

#         # ----------------------
#         # Step 2: Disease TFLite detection
#         # ----------------------
#         disease_result = run_tflite(image_np)
        
#         # ... (Rest of your code) ...
        
#         return jsonify({
#             "success": True,
#             "leaf_detected": True,
#             "disease": disease_result["disease"],
#             "confidence": disease_result["confidence"],
#             "raw_predictions": disease_result["raw"]
#         })

#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)}), 500
        
        


# @app.route('/health')
# def health():
#     return {
#         "status": "running",
#         "yolo_loaded": True,
#         "tflite_loaded": True
#     }


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io

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
# NEW HELPER: Leaf Shape Analysis (The Missing Part)
# ----------------------------------------------------
def is_likely_tomato_structure(image_np):
    """
    Checks if the leaf structure resembles a compound tomato leaf (low solidity)
    vs a solid hibiscus/rose leaf (high solidity).
    """
    try:
        # 1. Convert to Grayscale & Blur
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Threshold to get a binary mask of the leaf
        # Otsu's method finds the best threshold automatically
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False # No leaf found

        # 4. Get the largest contour (the main leaf)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # 5. Calculate Convex Hull & Solidity
        # Hull is the "rubber band" stretched around the object
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0: 
            return False

        # Solidity = Contour Area / Hull Area
        # Solid leaves (Hibiscus/Rose) have solidity > 0.9
        # Compound leaves (Tomato) have solidity < 0.85 (due to gaps between leaflets)
        solidity = area / hull_area
        
        print(f"DEBUG: Detected Solidity: {solidity}") # Check your terminal/logs

        # If it's too solid, it's likely NOT a tomato leaf
        # Adjust this 0.88 value if real tomato leaves get blocked
        if solidity > 0.88: 
            return False 
            
        return True
    except Exception as e:
        print(f"Error in shape check: {e}")
        return True # If check fails, default to allowing the image

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
        # Step 1.5: Shape/Solidity Check (New)
        # ----------------------
        if not is_likely_tomato_structure(image_np):
             return jsonify({
                "success": True,
                "leaf_detected": False, # Block it
                "message": "Leaf detected, but shape resembles Hibiscus/Rose (Solid Leaf) rather than Tomato."
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
