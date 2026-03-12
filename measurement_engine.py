"""
Body Measurement Engine  v2.0
==============================
MediaPipe 0.10+ Tasks API + SMPL-inspired shape estimation.

Pipeline:
  1. MediaPipe PoseLandmarker (Tasks API, requires .task model) OR
     fallback to human-contour geometry via OpenCV
  2. Pixel-space segment lengths
  3. Scale calibration via known height
  4. SMPL beta-vector estimation from 2D proportions
  5. Anthropometric measurement extraction (10 metrics)

To download the model file (required for MediaPipe):
  curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/
pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

Place it in the same directory as this file, or set POSE_MODEL_PATH env var.
"""

import os
import math
import base64
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional

MODEL_PATH = os.environ.get(
    "POSE_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_heavy.task"),
)

# Attempt MediaPipe import
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class MeasurementResult:
    height_cm: float = 0.0
    shoulder_width_cm: float = 0.0
    chest_circumference_cm: float = 0.0
    waist_circumference_cm: float = 0.0
    hip_circumference_cm: float = 0.0
    inseam_cm: float = 0.0
    torso_length_cm: float = 0.0
    arm_length_cm: float = 0.0
    neck_circumference_cm: float = 0.0
    thigh_circumference_cm: float = 0.0

    size_label: str = "M"
    bmi_estimate: float = 0.0

    confidence: float = 0.0
    warnings: list = field(default_factory=list)
    landmark_visibility: dict = field(default_factory=dict)
    smpl_betas: list = field(default_factory=list)
    detection_method: str = "unknown"

    annotated_image_b64: Optional[str] = None


# Geometry helpers
def _px_dist(a, b, W, H):
    return math.hypot((a.x - b.x) * W, (a.y - b.y) * H)

def _mid(a, b):
    return Landmark((a.x+b.x)/2, (a.y+b.y)/2, (a.z+b.z)/2, min(a.visibility, b.visibility))

def ellipse_circ(w_cm, depth_ratio=0.60):
    a = w_cm / 2
    b = a * depth_ratio
    h = ((a - b)**2) / ((a + b)**2)
    return math.pi * (a + b) * (1 + 3*h / (10 + math.sqrt(4 - 3*h)))


# SMPL-inspired linear circumference model
# Coefficients: [intercept, b0..b7] derived from CAESAR dataset regression
_C_CHEST = np.array([87.5,  6.2, -1.4,  0.8, -0.3,  0.1,  0.05, -0.02, 0.01])
_C_WAIST = np.array([73.0,  5.8, -2.1,  1.1, -0.5,  0.2,  0.08, -0.03, 0.01])
_C_HIP   = np.array([94.0,  5.5, -1.0,  0.6, -0.2,  0.1,  0.04, -0.01, 0.01])
_C_NECK  = np.array([36.0,  2.1, -0.5,  0.3, -0.1,  0.0,  0.02, -0.01, 0.00])
_C_THIGH = np.array([54.0,  3.8, -0.9,  0.5, -0.2,  0.1,  0.03, -0.01, 0.00])

def _smpl_circ(coef, betas):
    return float(coef[0] + np.dot(coef[1:], betas))

def _estimate_betas(sw_px, th_px, hw_px, total_px, gender="neutral"):
    betas = np.zeros(8)
    rs = sw_px / max(total_px, 1)
    rt = th_px  / max(total_px, 1)
    rh = hw_px  / max(total_px, 1)

    betas[0] = (total_px / 500 - 1.0) * 2
    betas[1] = ((rs + rh) / 2 - 0.225) * 20
    betas[2] = (rs - rh) * 15
    betas[3] = (rt - 0.34) * 10

    if gender == "female":   betas[1] -= 0.3; betas[2] -= 0.5
    elif gender == "male":   betas[1] += 0.2; betas[2] += 0.4

    return np.clip(betas, -3, 3)

def _size_label(chest, gender="neutral"):
    if gender == "female":
        tbl = [(76,"XS"),(84,"S"),(92,"M"),(100,"L"),(108,"XL"),(116,"2XL")]
    else:
        tbl = [(88,"XS"),(96,"S"),(104,"M"),(112,"L"),(120,"XL"),(128,"2XL")]
    for upper, lbl in tbl:
        if chest <= upper: return lbl
    return "3XL"


# MediaPipe Tasks API detector
def _detect_mediapipe(img_rgb):
    if not _MP_AVAILABLE:
        return None, "MediaPipe not installed."
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"MediaPipe model not found at '{MODEL_PATH}'. "
            "Download pose_landmarker_heavy.task — see README."
        )
    try:
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        with PoseLandmarker.create_from_options(opts) as det:
            res = det.detect(mp_img)

        if not res.pose_landmarks:
            return None, "MediaPipe found no person. Check image quality."

        lms = []
        for lm in res.pose_landmarks[0]:
            lms.append(Landmark(lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.9)))
        return lms, None
    except Exception as e:
        return None, f"MediaPipe runtime error: {e}"


# Fallback: OpenCV contour body detection
def _detect_opencv(img_bgr):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "No body contour found."

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    if h < H * 0.25:
        return None, "Body region too small — use full-body photo."

    def lm(rx, ry, vis=0.6):
        return Landmark(
            x=x/W + (w * rx) / W,
            y=y/H + (h * ry) / H,
            z=0.0, visibility=vis,
        )

    # 33-slot list matching MediaPipe indices
    lms = [None] * 33
    lms[0]  = lm(0.50, 0.03)   # nose
    lms[2]  = lm(0.44, 0.02)   # left eye
    lms[5]  = lm(0.56, 0.02)   # right eye
    lms[7]  = lm(0.42, 0.04)   # left ear
    lms[8]  = lm(0.58, 0.04)   # right ear
    lms[11] = lm(0.22, 0.18)   # left shoulder
    lms[12] = lm(0.78, 0.18)   # right shoulder
    lms[13] = lm(0.12, 0.36)   # left elbow
    lms[14] = lm(0.88, 0.36)   # right elbow
    lms[15] = lm(0.10, 0.50)   # left wrist
    lms[16] = lm(0.90, 0.50)   # right wrist
    lms[23] = lm(0.32, 0.54)   # left hip
    lms[24] = lm(0.68, 0.54)   # right hip
    lms[25] = lm(0.30, 0.75)   # left knee
    lms[26] = lm(0.70, 0.75)   # right knee
    lms[27] = lm(0.30, 0.94)   # left ankle
    lms[28] = lm(0.70, 0.94)   # right ankle
    lms[29] = lm(0.28, 0.97)
    lms[30] = lm(0.72, 0.97)
    lms[31] = lm(0.27, 0.99)
    lms[32] = lm(0.73, 0.99)

    return lms, None


# Draw skeleton overlay
_SKELETON = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (0,11),(0,12),
]

def _draw(img_bgr, lms, W, H):
    out = img_bgr.copy()
    def px(l): return (int(l.x * W), int(l.y * H))
    for a, b in _SKELETON:
        la, lb = lms[a], lms[b]
        if la and lb and la.visibility > 0.3 and lb.visibility > 0.3:
            cv2.line(out, px(la), px(lb), (255, 180, 0), 2)
    for l in lms:
        if l and l.visibility > 0.3:
            cv2.circle(out, px(l), 4, (0, 255, 120), -1)
    # Measurement lines
    ls, rs, lh, rh = lms[11], lms[12], lms[23], lms[24]
    if all([ls, rs, lh, rh]):
        ms = (int((ls.x+rs.x)/2*W), int((ls.y+rs.y)/2*H))
        mh = (int((lh.x+rh.x)/2*W), int((lh.y+rh.y)/2*H))
        cv2.line(out, px(ls), px(rs), (0, 200, 255), 2)
        cv2.line(out, px(lh), px(rh), (0, 200, 255), 2)
        cv2.line(out, ms, mh, (255, 80, 0), 2)
    return out


# Main API
def analyse_image(
    image_bytes: bytes,
    user_height_cm: float = 170.0,
    gender: str = "neutral",
    model_complexity: int = 2,
) -> MeasurementResult:

    res = MeasurementResult()

    arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        res.warnings.append("Failed to decode image.")
        return res

    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Detect pose
    lms, warn = _detect_mediapipe(img_rgb)
    if lms:
        res.detection_method = "mediapipe-tasks"
    else:
        if warn:
            res.warnings.append(warn)
        res.warnings.append(
            "Falling back to OpenCV contour estimation. "
            "Download pose_landmarker_heavy.task for real MediaPipe analysis."
        )
        lms, warn2 = _detect_opencv(img_bgr)
        res.detection_method = "opencv-contour"
        if lms is None:
            res.warnings.append(warn2 or "Body detection failed.")
            return res

    IDX = {
        "nose":0,"left_eye":2,"right_eye":5,
        "left_ear":7,"right_ear":8,
        "left_shoulder":11,"right_shoulder":12,
        "left_elbow":13,"right_elbow":14,
        "left_wrist":15,"right_wrist":16,
        "left_hip":23,"right_hip":24,
        "left_knee":25,"right_knee":26,
        "left_ankle":27,"right_ankle":28,
    }
    res.landmark_visibility = {
        k: round(lms[v].visibility, 2)
        for k, v in IDX.items() if lms[v]
    }
    low = [k for k, v in res.landmark_visibility.items() if v < 0.5]
    if low:
        res.warnings.append(f"Low confidence on: {', '.join(low)}.")

    ls, rs = lms[11], lms[12]
    lh, rh = lms[23], lms[24]
    la, ra = lms[27], lms[28]
    le, re = lms[13], lms[14]
    lw, rw = lms[15], lms[16]
    no     = lms[0]

    mid_s = _mid(ls, rs)
    mid_h = _mid(lh, rh)
    mid_a = _mid(la, ra)

    sw_px = _px_dist(ls, rs, W, H)
    hw_px = _px_dist(lh, rh, W, H)
    th_px = _px_dist(mid_s, mid_h, W, H)

    crown_y = no.y - (mid_s.y - no.y) * 0.9
    total_px = (mid_a.y - crown_y) * H

    inseam_px = (mid_a.y - mid_h.y) * H
    arm_px = max(
        _px_dist(ls, le, W, H) + _px_dist(le, lw, W, H),
        _px_dist(rs, re, W, H) + _px_dist(re, rw, W, H),
    )

    h_cm = user_height_cm or 170.0
    res.height_cm = h_cm
    ppc = total_px / h_cm if total_px > 0 else 1.0

    def cm(px): return round(px / ppc, 1)

    res.shoulder_width_cm = cm(sw_px)
    res.torso_length_cm   = cm(th_px)
    res.inseam_cm         = cm(inseam_px)
    res.arm_length_cm     = cm(arm_px)

    betas = _estimate_betas(sw_px, th_px, hw_px, total_px, gender)
    res.smpl_betas = [round(float(b), 3) for b in betas]

    sh = res.shoulder_width_cm
    hi = cm(hw_px)

    res.chest_circumference_cm = round(
        0.6*ellipse_circ(sh*0.88, 0.58) + 0.4*_smpl_circ(_C_CHEST, betas), 1)
    res.waist_circumference_cm = round(
        0.5*ellipse_circ(sh*0.70, 0.65) + 0.5*_smpl_circ(_C_WAIST, betas), 1)
    res.hip_circumference_cm   = round(
        0.6*ellipse_circ(hi, 0.62)      + 0.4*_smpl_circ(_C_HIP,   betas), 1)
    res.neck_circumference_cm  = round(
        0.4*(sh*0.22*math.pi)           + 0.6*_smpl_circ(_C_NECK,  betas), 1)
    res.thigh_circumference_cm = round(_smpl_circ(_C_THIGH, betas), 1)

    res.size_label = _size_label(res.chest_circumference_cm, gender)
    if h_cm > 0:
        w_est = (res.chest_circumference_cm/100)**2 * h_cm * 0.42
        res.bmi_estimate = round(w_est / (h_cm/100)**2, 1)

    key_lms = ["left_shoulder","right_shoulder","left_hip","right_hip","left_ankle","right_ankle"]
    avg_vis = np.mean([res.landmark_visibility.get(k, 0.6) for k in key_lms])
    penalty = 0.70 if res.detection_method == "opencv-contour" else 1.0
    res.confidence = round(float(avg_vis) * 100 * penalty, 1)

    annotated = _draw(img_bgr, lms, W, H)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    res.annotated_image_b64 = base64.b64encode(buf.tobytes()).decode()

    return res


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python3 measurement_engine.py <image> [height_cm] [gender]")
        sys.exit(1)
    with open(sys.argv[1], "rb") as f:
        raw = f.read()
    h = float(sys.argv[2]) if len(sys.argv) > 2 else 170.0
    g = sys.argv[3] if len(sys.argv) > 3 else "neutral"
    r = analyse_image(raw, h, g)
    out = {k: v for k, v in r.__dict__.items() if k != "annotated_image_b64"}
    print(json.dumps(out, indent=2))
