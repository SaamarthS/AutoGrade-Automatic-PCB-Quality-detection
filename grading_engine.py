from ultralytics import YOLO
import cv2
import numpy as np

# ── Severity weights — updated for combined model class names ──────────────
SEVERITY_WEIGHTS = {
    # ── YOLO model classes ─────────────────────────────
    'open_circuit':        1.00,
    'short':               0.90,
    'mouse_bite':          0.70,
    'missing_hole':        0.70,
    'spur':                0.25,
    'spurious_copper':     0.25,
    # ── Groq vision — solder defect classes ───────────
    # (severity set by PCB engineering judgment)
    'solder_bridge':       0.88,  # direct short between pads
    'lifted_pad':          0.82,  # permanent board damage
    'tombstone':           0.80,  # component open on one end
    'solder_burn':         0.75,  # thermal damage / overheating
    'cold_joint':          0.70,  # intermittent / unreliable connection
    'solder_ball':         0.65,  # free particle, latent short risk
    'insufficient_solder': 0.60,  # weak mechanical+electrical joint
    'excess_solder':       0.50,  # may bridge adjacent pads
    'solder_defect':       0.65,  # generic catch-all
    'other':               0.55,
}

# ── Grade thresholds ───────────────────────────────────────────────────────
GRADE_THRESHOLDS = {
    'A': 90,
    'B': 75,
    'C': 60,
}

def load_model(model_path='best.pt'):
    return YOLO(model_path)

def grade_pcb(image_path, model, conf=0.45, iou=0.4):
    """
    Takes a PCB image path, runs detection,
    computes quality score and returns full breakdown.
    """

    # ── Run detection ──────────────────────────────────────────────────────
    results = model.predict(source=image_path, conf=conf, iou=iou, verbose=False)
    result  = results[0]

    # ── Image dimensions ───────────────────────────────────────────────────
    h, w       = result.orig_shape
    board_area = h * w

    # ── Class-specific confidence thresholds ────────────────────────────────
    CLASS_CONF_THRESHOLD = {
        'missing_hole':    0.65,
        'open_circuit':    0.40,
        'short':           0.40,
        'mouse_bite':      0.45,
        'spur':             0.40,
        'spurious_copper': 0.40,
    }

    # ── Parse detections ───────────────────────────────────────────────────
    defects = []
    for box in result.boxes:
        class_id   = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        min_conf = CLASS_CONF_THRESHOLD.get(class_name, 0.45)
        if confidence < min_conf:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        defect_area = (x2 - x1) * (y2 - y1)

        defects.append({
            'class':      class_name,
            'confidence': confidence,
            'area':       defect_area,
            'area_pct':   round((defect_area / board_area) * 100, 3),
            'bbox':       (x1, y1, x2, y2),
            'severity':   SEVERITY_WEIGHTS.get(class_name, 0.50),
            'zone':       '—'   # filled in later during zone calculation
        })

    # ── Step 1: Per-defect contribution ───────────────────────────────────
    d_scores = []
    for d in defects:
        w_i    = d['severity']
        conf_i = d['confidence']
        area_i = d['area'] / board_area
        d_i    = w_i * conf_i * area_i
        d_scores.append(d_i)

    # ── Step 2: Aggregate defect score (capped at 1.0) ────────────────────
    D = min(sum(d_scores), 1.0)

    # ── Step 3: Defect density ────────────────────────────────────────────
    N   = len(defects)
    rho = N / (board_area / 10000)

    # ── Step 4: Cleanliness percentage ────────────────────────────────────
    total_defect_area = sum(d['area'] for d in defects)
    C = (1 - total_defect_area / board_area) * 100

    # ── Step 5: Zone penalty (centre = critical zone) ─────────────────────
    cx, cy = w / 2, h / 2
    Z = 0
    for d, d_i in zip(defects, d_scores):
        x1, y1, x2, y2 = d['bbox']
        defect_cx = (x1 + x2) / 2
        defect_cy = (y1 + y2) / 2
        in_critical = (abs(defect_cx - cx) < w * 0.33 and
                       abs(defect_cy - cy) < h * 0.33)
        if in_critical:
            Z += d_i * 1.5
            d['zone'] = 'critical'
        else:
            d['zone'] = 'normal'

    # ── Step 6: Final score ───────────────────────────────────────────────
    score = 100 - (D * 40) - (min(rho, 1.0) * 30) - (min(Z, 1.0) * 30)
    score = max(0, min(100, round(score, 2)))

    # ── Grade assignment ──────────────────────────────────────────────────
    if score >= GRADE_THRESHOLDS['A']:
        grade = 'A'
    elif score >= GRADE_THRESHOLDS['B']:
        grade = 'B'
    elif score >= GRADE_THRESHOLDS['C']:
        grade = 'C'
    else:
        grade = 'Reject'

    return {
        'score':        score,
        'grade':        grade,
        'defects':      defects,
        'defect_count': N,
        'cleanliness':  round(C, 2),
        'D':            round(D, 4),
        'rho':          round(rho, 4),
        'Z':            round(Z, 4),
    }

def get_annotated_image(image_path, model, conf=0.45, iou=0.4):
    """Returns annotated image as numpy array for display."""
    results = model.predict(source=image_path, conf=conf, iou=iou, verbose=False)
    return results[0].plot()


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = load_model('best.pt')
    print("Classes:", model.names)
    print("✅ Model loaded successfully!")
    print("Grading engine ready.")