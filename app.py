"""
Flask API Server — Body Measurement Service
===========================================
Endpoints:
  POST /api/measure   — analyse uploaded image → JSON measurements
  GET  /api/health    — liveness probe
  GET  /api/size-guide — standard size chart

Run:
  python3 app.py
"""

import os
import json
import base64
from flask import Flask, request, jsonify
from measurement_engine import analyse_image, MeasurementResult

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# ── CORS (manual, no flask-cors dependency) ───────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/api/measure", methods=["OPTIONS"])
def measure_preflight():
    return "", 204


# ── Measurement endpoint ──────────────────────────────────────────────────────
@app.route("/api/measure", methods=["POST"])
def measure():
    """
    Accept multipart/form-data with:
      - file        : image (JPEG/PNG)
      - height_cm   : (optional) float, user's known height
      - gender      : (optional) "male" | "female" | "neutral"
      - complexity  : (optional) 0|1|2, MediaPipe model complexity
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form-data key 'file'."}), 400

    img_file = request.files["file"]
    if img_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Validate type
    allowed = {"image/jpeg", "image/png", "image/webp"}
    ct = img_file.content_type or ""
    if ct not in allowed and not img_file.filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    ):
        return jsonify({"error": f"Unsupported image type: {ct}"}), 415

    image_bytes = img_file.read()

    # Parse optional params
    try:
        height_cm = float(request.form.get("height_cm", 170))
    except ValueError:
        height_cm = 170.0

    gender     = request.form.get("gender", "neutral").lower()
    if gender not in ("male", "female", "neutral"):
        gender = "neutral"

    try:
        complexity = int(request.form.get("complexity", 2))
        complexity = max(0, min(2, complexity))
    except ValueError:
        complexity = 2

    # Run analysis
    try:
        result: MeasurementResult = analyse_image(
            image_bytes,
            user_height_cm=height_cm,
            gender=gender,
            model_complexity=complexity,
        )
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {str(exc)}"}), 500

    # Serialise
    payload = {
        "success": result.confidence > 0,
        "measurements": {
            "height_cm":              result.height_cm,
            "shoulder_width_cm":      result.shoulder_width_cm,
            "chest_circumference_cm": result.chest_circumference_cm,
            "waist_circumference_cm": result.waist_circumference_cm,
            "hip_circumference_cm":   result.hip_circumference_cm,
            "inseam_cm":              result.inseam_cm,
            "torso_length_cm":        result.torso_length_cm,
            "arm_length_cm":          result.arm_length_cm,
            "neck_circumference_cm":  result.neck_circumference_cm,
            "thigh_circumference_cm": result.thigh_circumference_cm,
        },
        "sizing": {
            "recommended_size": result.size_label,
            "bmi_estimate":     result.bmi_estimate,
        },
        "smpl_betas":            result.smpl_betas,
        "confidence_pct":        result.confidence,
        "warnings":              result.warnings,
        "landmark_visibility":   result.landmark_visibility,
        "annotated_image_b64":   result.annotated_image_b64,
    }

    return jsonify(payload), 200


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "MediaPipe+SMPL-approx v1.0"}), 200


# ── Size guide ────────────────────────────────────────────────────────────────
SIZE_GUIDE = {
    "male": [
        {"size": "XS",  "chest": "80-88",  "waist": "64-72",  "hip": "84-92"},
        {"size": "S",   "chest": "88-96",  "waist": "72-80",  "hip": "92-100"},
        {"size": "M",   "chest": "96-104", "waist": "80-88",  "hip": "100-108"},
        {"size": "L",   "chest": "104-112","waist": "88-96",  "hip": "108-116"},
        {"size": "XL",  "chest": "112-120","waist": "96-104", "hip": "116-124"},
        {"size": "2XL", "chest": "120-128","waist": "104-112","hip": "124-132"},
        {"size": "3XL", "chest": "128+",   "waist": "112+",   "hip": "132+"},
    ],
    "female": [
        {"size": "XS",  "chest": "76-84",  "waist": "60-68",  "hip": "84-92"},
        {"size": "S",   "chest": "84-92",  "waist": "68-76",  "hip": "92-100"},
        {"size": "M",   "chest": "92-100", "waist": "76-84",  "hip": "100-108"},
        {"size": "L",   "chest": "100-108","waist": "84-92",  "hip": "108-116"},
        {"size": "XL",  "chest": "108-116","waist": "92-100", "hip": "116-124"},
        {"size": "2XL", "chest": "116+",   "waist": "100+",   "hip": "124+"},
    ],
}

@app.route("/api/size-guide", methods=["GET"])
def size_guide():
    gender = request.args.get("gender", "male").lower()
    guide  = SIZE_GUIDE.get(gender, SIZE_GUIDE["male"])
    return jsonify({"gender": gender, "unit": "cm", "guide": guide}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n🚀 Body Measurement API running on http://localhost:{port}")
    print("   POST /api/measure   — analyse a photo")
    print("   GET  /api/health    — health check")
    print("   GET  /api/size-guide — size chart\n")
    app.run(host="0.0.0.0", port=port, debug=False)
