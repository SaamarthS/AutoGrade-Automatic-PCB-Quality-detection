"""
AutoGrade — Batch Evaluation Script
=====================================
Runs inference on every image in PCB/, saves annotated output images,
and produces a CSV report with detection results + a ground-truth column
you can fill in manually to compute accuracy metrics.

Usage:
    python batch_eval.py                         # default settings
    python batch_eval.py --conf 0.35             # lower threshold
    python batch_eval.py --preprocess            # enable preprocessing
    python batch_eval.py --model DeepPCB.pt      # use alternate model
    python batch_eval.py --metrics               # compute metrics (after filling gt_grade in CSV)

Outputs:
    output/annotated/    — annotated images for every PCB
    output/results.csv   — per-image results table
    output/metrics.txt   — accuracy report (only with --metrics flag)
"""

import os
import cv2
import argparse
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from grading_engine import load_model, grade_pcb, SEVERITY_WEIGHTS

# ── Class colours (consistent with app.py) ─────────────────────────────────
CLASS_COLORS = {
    'open_circuit':    (255, 50,  50),
    'short':           (50,  50,  255),
    'mouse_bite':      (50,  200, 50),
    'spur':            (255, 165, 0),
    'spurious_copper': (180, 0,   255),
    'missing_hole':    (0,   200, 220),
}

CLASS_CONF_THRESHOLD = {
    'missing_hole':    0.65,
    'open_circuit':    0.40,
    'short':           0.40,
    'mouse_bite':      0.45,
    'spur':            0.40,
    'spurious_copper': 0.40,
}


# ── Preprocessing (mirrors app.py) ─────────────────────────────────────────
def preprocess_pcb(img_bgr, clahe_clip=2.5, clahe_grid=8,
                   sharpen_strength=1.5, denoise_h=5):
    # CLAHE
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Unsharp mask
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    img = cv2.addWeighted(img, 1 + sharpen_strength, blurred, -sharpen_strength, 0)

    # NL-means denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, h=denoise_h, hColor=denoise_h,
                                           templateWindowSize=7, searchWindowSize=21)
    return img


# ── Annotated image ─────────────────────────────────────────────────────────
def annotate_image(img_bgr, model, conf=0.45, iou=0.4):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        tmp_path = tmp.name

    results = model.predict(source=tmp_path, conf=conf, iou=iou, verbose=False)
    result  = results[0]
    CLASSES = result.names

    img = img_bgr.copy()

    for box in result.boxes:
        cls_id   = int(box.cls[0])
        cls_name = CLASSES[cls_id]
        conf_val = float(box.conf[0])

        if conf_val < CLASS_CONF_THRESHOLD.get(cls_name, 0.45):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(cls_name, (100, 100, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf_val:.2f}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs    = 0.42
        th    = 1
        (tw, text_h), _ = cv2.getTextSize(label, font, fs, th)
        lx = max(0, x1)
        ly = max(text_h + 4, y1 - 4)
        cv2.rectangle(img, (lx, ly - text_h - 4), (lx + tw + 4, ly + 2), color, -1)
        cv2.putText(img, label, (lx + 2, ly - 2), font, fs, (255, 255, 255), th, cv2.LINE_AA)

    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Metrics calculator ──────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> str:
    """
    Compares 'grade' (model output) vs 'gt_grade' (your manual labels).
    Prints per-class precision, recall, overall accuracy.
    """
    filled = df[df['gt_grade'].notna() & (df['gt_grade'].str.strip() != '')]
    if filled.empty:
        return "⚠️  No ground truth data found. Fill the 'gt_grade' column in results.csv first."

    total   = len(filled)
    correct = (filled['grade'] == filled['gt_grade']).sum()
    acc     = correct / total * 100

    grades  = ['A', 'B', 'C', 'Reject']
    lines   = [
        "=" * 52,
        f"  AutoGrade — Batch Evaluation Metrics",
        f"  Generated: {datetime.datetime.now().strftime('%d %b %Y %H:%M')}",
        "=" * 52,
        f"  Images evaluated : {total}",
        f"  Overall accuracy : {acc:.1f}%  ({correct}/{total} correct)",
        "-" * 52,
        f"  {'Grade':<10} {'Precision':>12} {'Recall':>10} {'Support':>10}",
        "-" * 52,
    ]

    for g in grades:
        tp = ((filled['grade'] == g) & (filled['gt_grade'] == g)).sum()
        fp = ((filled['grade'] == g) & (filled['gt_grade'] != g)).sum()
        fn = ((filled['grade'] != g) & (filled['gt_grade'] == g)).sum()
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        supp = (filled['gt_grade'] == g).sum()
        lines.append(f"  {g:<10} {prec:>11.1f}% {rec:>9.1f}% {supp:>10}")

    lines += [
        "-" * 52,
        "",
        "  Confusion matrix (rows=actual, cols=predicted):",
        "",
    ]

    # Confusion matrix
    header = f"  {'':>8}" + "".join(f"{g:>10}" for g in grades)
    lines.append(header)
    for actual in grades:
        row = f"  {actual:>8}"
        for pred in grades:
            n = ((filled['gt_grade'] == actual) & (filled['grade'] == pred)).sum()
            row += f"{n:>10}"
        lines.append(row)

    lines.append("=" * 52)
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AutoGrade Batch Evaluator")
    parser.add_argument('--input',      default='PCB',       help='Folder with PCB images')
    parser.add_argument('--output',     default='output',    help='Output folder')
    parser.add_argument('--model',      default='best.pt',   help='Model weights file')
    parser.add_argument('--conf',       type=float, default=0.45)
    parser.add_argument('--iou',        type=float, default=0.40)
    parser.add_argument('--preprocess', action='store_true', help='Enable image preprocessing')
    parser.add_argument('--metrics',    action='store_true', help='Compute metrics from existing results.csv')
    args = parser.parse_args()

    out_dir  = Path(args.output)
    ann_dir  = out_dir / 'annotated'
    csv_path = out_dir / 'results.csv'
    met_path = out_dir / 'metrics.txt'

    # ── Metrics-only mode ───────────────────────────────────────────────────
    if args.metrics:
        if not csv_path.exists():
            print(f"❌ results.csv not found at {csv_path}. Run without --metrics first.")
            return
        df     = pd.read_csv(csv_path)
        report = compute_metrics(df)
        print(report)
        met_path.write_text(report)
        print(f"\n✅ Metrics saved to {met_path}")
        return

    # ── Setup ───────────────────────────────────────────────────────────────
    ann_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔬 AutoGrade Batch Evaluator")
    print(f"   Model      : {args.model}")
    print(f"   Input      : {args.input}/")
    print(f"   Output     : {args.output}/")
    print(f"   Confidence : {args.conf}")
    print(f"   IoU        : {args.iou}")
    print(f"   Preprocess : {'ON' if args.preprocess else 'OFF'}")
    print()

    model = load_model(args.model)

    # ── Collect images ───────────────────────────────────────────────────────
    input_dir = Path(args.input)
    exts      = {'.jpg', '.jpeg', '.png', '.bmp'}
    images    = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])

    if not images:
        print(f"❌ No images found in {args.input}/")
        return

    print(f"📂 Found {len(images)} images\n")

    rows = []

    for img_path in tqdm(images, desc="Processing", unit="img"):
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                img_bgr = np.array(Image.open(img_path).convert('RGB'))
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

            # Preprocessing
            proc_bgr = preprocess_pcb(img_bgr) if args.preprocess else img_bgr

            # Save temp file for grading engine
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                cv2.imwrite(tmp.name, proc_bgr)
                tmp_path = tmp.name

            # Grade
            result = grade_pcb(tmp_path, model, conf=args.conf, iou=args.iou)

            # Annotate & save
            ann_rgb  = annotate_image(proc_bgr, model, conf=args.conf, iou=args.iou)
            ann_bgr  = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)
            ann_out  = ann_dir / img_path.name
            cv2.imwrite(str(ann_out), ann_bgr)

            # Defect class summary
            defect_classes = ', '.join(set(d['class'] for d in result['defects'])) or 'none'

            rows.append({
                'filename':      img_path.name,
                'grade':         result['grade'],
                'score':         result['score'],
                'defect_count':  result['defect_count'],
                'defect_classes': defect_classes,
                'cleanliness':   result['cleanliness'],
                'D':             result['D'],
                'rho':           result['rho'],
                'Z':             result['Z'],
                'gt_grade':      '',   # ← YOU FILL THIS IN (A / B / C / Reject)
                'notes':         '',   # ← optional notes column
            })

            os.unlink(tmp_path)

        except Exception as e:
            rows.append({
                'filename':      img_path.name,
                'grade':         'ERROR',
                'score':         -1,
                'defect_count':  -1,
                'defect_classes': str(e),
                'cleanliness':   -1,
                'D': -1, 'rho': -1, 'Z': -1,
                'gt_grade': '', 'notes': '',
            })

    # ── Save CSV ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = df[df['grade'] != 'ERROR']
    grade_counts = valid['grade'].value_counts().to_dict()

    print(f"\n{'='*48}")
    print(f"  ✅ Done! Processed {len(df)} images")
    print(f"  Grade distribution:")
    for g in ['A', 'B', 'C', 'Reject']:
        bar = '█' * grade_counts.get(g, 0)
        print(f"    {g:<8} {grade_counts.get(g, 0):>4}  {bar}")
    print(f"\n  Annotated images → {ann_dir}/")
    print(f"  Results CSV      → {csv_path}")
    print(f"\n  📝 Next step: open results.csv and fill in the")
    print(f"     'gt_grade' column with your expert grades,")
    print(f"     then run:  python batch_eval.py --metrics")
    print(f"{'='*48}\n")


if __name__ == '__main__':
    main()
