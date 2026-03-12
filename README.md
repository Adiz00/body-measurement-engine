# BodyFit.ai — AI Body Measurement Engine
### Final Year Project · AI-Powered E-Commerce Platform

---

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │         React Frontend           │
                    │   (frontend.jsx → Vite / CRA)   │
                    └──────────────┬──────────────────┘
                                   │ POST /api/measure
                                   │ (multipart/form-data)
                    ┌──────────────▼──────────────────┐
                    │       Flask API  (app.py)        │
                    │       localhost:5001             │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   measurement_engine.py          │
                    │                                  │
                    │  Step 1: MediaPipe Pose v2       │
                    │    → 33 3-D landmarks            │
                    │                                  │
                    │  Step 2: Pixel geometry          │
                    │    → lengths in image units      │
                    │                                  │
                    │  Step 3: Scale calibration       │
                    │    → cm via known height         │
                    │                                  │
                    │  Step 4: SMPL β estimation       │
                    │    → shape parameters            │
                    │                                  │
                    │  Step 5: Measurement extraction  │
                    │    → 10 anthropometric values    │
                    └─────────────────────────────────┘
```

---

## Pipeline Detail

### Step 1 — MediaPipe Pose Estimation
- Model: `pose_landmarker_heavy.task` (complexity=2)
- Output: 33 landmarks, each with normalised (x, y, z) + visibility
- z-coordinate is relative depth in MediaPipe's body-height units

### Step 2 — Pixel-Space Geometry
Key measurements extracted directly:
| Measurement | Landmarks used |
|---|---|
| Shoulder width | Left shoulder ↔ Right shoulder |
| Torso length | Shoulder midpoint ↔ Hip midpoint |
| Inseam | Hip midpoint ↔ Ankle midpoint |
| Arm length | Shoulder → Elbow → Wrist (each side) |
| Hip width | Left hip ↔ Right hip |

### Step 3 — Scale Calibration
```
px_per_cm = total_body_height_px / user_height_cm
measurement_cm = measurement_px / px_per_cm
```
Without user height: defaults to 170 cm (±5-10 cm error introduced).

### Step 4 — SMPL β Shape Parameter Estimation
Simplified 9-parameter β-vector estimated from 2D proportions:
- β₀: height deviation from average
- β₁: adiposity proxy (body width / height ratio)
- β₂: shoulder-to-hip ratio (somatotype)
- β₃: torso length ratio
- β₄–β₈: residual variation (near zero for frontal photo)

Real SMPL uses 300 components from PCA on 4,000+ scans (CAESAR dataset).
Our simplified model uses published regression coefficients.

### Step 5 — Circumference Estimation
Circumferences can't be read directly from a frontal photo.
We use a **blended model**:
- **Geometric** (60% weight): Ramanujan's ellipse approximation
  - `C ≈ π × (a + b) × [1 + 3h/(10 + √(4-3h))]`
  - where a = frontal half-width, b = a × depth_ratio (0.55–0.65)
- **SMPL-linear** (40% weight): `C = coef₀ + β·coef₁₋₉`
  - Coefficients derived from CAESAR anthropometric dataset

---

## Measurements Output

| Field | Unit | Method |
|---|---|---|
| height_cm | cm | user-provided / estimated |
| shoulder_width_cm | cm | direct landmark distance |
| chest_circumference_cm | cm | geometric + SMPL blend |
| waist_circumference_cm | cm | geometric + SMPL blend |
| hip_circumference_cm | cm | geometric + SMPL blend |
| inseam_cm | cm | direct landmark distance |
| torso_length_cm | cm | direct landmark distance |
| arm_length_cm | cm | shoulder→elbow→wrist path |
| neck_circumference_cm | cm | SMPL-linear (estimated) |
| thigh_circumference_cm | cm | SMPL-linear (estimated) |

---

## Setup & Run

### Backend (Python 3.9+)
```bash
pip install flask mediapipe opencv-python-headless numpy Pillow

python3 app.py
# → running on http://localhost:5001
```

### Frontend (React)
Option A — Embed `frontend.jsx` in your existing React app (default export).
Option B — Create with Vite:
```bash
npm create vite@latest frontend -- --template react
cd frontend
# copy frontend.jsx → src/App.jsx
npm install
npm run dev
```

### Test via curl
```bash
curl -X POST http://localhost:5001/api/measure \
  -F "file=@photo.jpg" \
  -F "height_cm=175" \
  -F "gender=male" | python3 -m json.tool
```

---

## API Reference

### POST /api/measure
**Form fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| file | image/* | ✓ | JPEG/PNG/WebP, max 16 MB |
| height_cm | float | recommended | Known height (improves accuracy) |
| gender | string | optional | male / female / neutral |
| complexity | int | optional | 0/1/2 (default 2 = best) |

**Response:**
```json
{
  "success": true,
  "measurements": {
    "height_cm": 175.0,
    "shoulder_width_cm": 43.2,
    "chest_circumference_cm": 98.4,
    "waist_circumference_cm": 82.1,
    "hip_circumference_cm": 97.6,
    "inseam_cm": 81.3,
    "torso_length_cm": 57.8,
    "arm_length_cm": 62.4,
    "neck_circumference_cm": 38.2,
    "thigh_circumference_cm": 54.7
  },
  "sizing": {
    "recommended_size": "M",
    "bmi_estimate": 22.4
  },
  "smpl_betas": [0.12, -0.34, 0.56, -0.11, 0.08, 0.03, -0.02, 0.01, 0.0],
  "confidence_pct": 78.3,
  "warnings": [],
  "landmark_visibility": { "left_shoulder": 0.98, "right_shoulder": 0.97, "..." },
  "annotated_image_b64": "<jpeg-base64>"
}
```

---

## Known Limitations & Future Work

| Limitation | Impact | Fix |
|---|---|---|
| Monocular depth ambiguity | ±3-7 cm on circumferences | Add side-view photo |
| Clothing occlusion | Landmark shifts | Segment body first (SAM) |
| No crown-of-head landmark | Height slightly underestimated | Add reference object |
| SMPL β is simplified | Circumference accuracy ±3-5 cm | Use full SMPLify-X pipeline |
| No camera calibration | Perspective distortion | Use known reference object |

### Recommended Upgrade Path (post-FYP)
1. **SMPLify-X** full pipeline (PyTorch) — real SMPL mesh → extract measurements via `smpl-anthropometry` library
2. **Multi-view** — front + side photo → 2× accuracy
3. **Reference object** — A4 paper / credit card → removes need for height input
4. **SnapMeasureAI / ReelSize API** — commercial option, 80+ measurements, 98% claimed accuracy

---

## Project Structure
```
body-measurement-app/
├── measurement_engine.py  ← Core pipeline (MediaPipe + SMPL approximation)
├── app.py                 ← Flask REST API
├── frontend.jsx           ← React UI component
└── README.md              ← This file
```
"# body-measurement-engine" 
