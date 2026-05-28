"""
groq_vision.py — Step 2 of the AutoGrade 2-step PCB inspection pipeline.

Responsibilities:
  1. is_pcb_image()         → Validates the image is actually a PCB before processing.
  2. detect_solder_defects() → Finds solder-specific defects (burns, bridges, cold joints, etc.)
                               that the YOLO model was NOT trained on, returns pixel-space bboxes.

Uses Groq's OpenAI-compatible API with a vision-capable model.
"""

import base64
import json
import re
import cv2
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
# Comma-separated keys in .env  e.g. GROQ_API_KEYS=key1,key2
def _load_groq_keys() -> list:
    """Load Groq API keys from env or st.secrets (Streamlit Cloud fallback)."""
    # 1. Try OS environment / .env file
    keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
    if keys:
        return keys
    # 2. Fallback: st.secrets (Streamlit Community Cloud)
    try:
        import streamlit as st
        raw = st.secrets.get("GROQ_API_KEYS", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
    except Exception:
        pass
    return keys

GROQ_API_KEYS = _load_groq_keys()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Vision-capable models, tried in order per API key.
# Llama-4 variants are preferred; legacy 3.2 are fallbacks.
GROQ_VISION_MODELS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.2-90b-vision-preview",
    "llama-3.2-11b-vision-preview",
]

# ── Solder defect severity weights (PCB engineering judgment) ────────────────
SOLDER_SEVERITY = {
    'solder_bridge':       0.88,   # direct short between pads
    'lifted_pad':          0.82,   # permanent board damage
    'solder_burn':         0.75,   # thermal damage / overheating
    'cold_joint':          0.70,   # intermittent unreliable connection
    'solder_ball':         0.65,   # free particle, latent short risk
    'tombstone':           0.80,   # component open on one end
    'insufficient_solder': 0.60,   # weak mechanical + electrical joint
    'excess_solder':       0.50,   # may bridge adjacent pads
    'solder_defect':       0.65,   # generic catch-all
    'other':               0.55,
}

# ── Hallucination guard thresholds ───────────────────────────────────────────
SOLDER_CONF_MIN    = 0.60   # discard detections below this confidence
BBOX_MAX_AREA_FRAC = 0.30   # discard boxes covering >30% of image (almost always hallucinated)
BBOX_MIN_AREA_FRAC = 0.0005 # discard sub-pixel boxes (<0.05% of image)
BBOX_MAX_SIDE_FRAC = 0.75   # discard boxes spanning >75% of image width OR height


# ── Helpers ──────────────────────────────────────────────────────────────────
def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _encode_image(image_path: str) -> str:
    """Encode image to base64 string for Groq API."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _try_models(messages: list, max_tokens: int = 512, temperature: float = 0.0) -> tuple:
    """
    Try every (api_key, model) combination until one succeeds.
    Returns (response_text, model_name_used).
    Raises RuntimeError if ALL combinations fail.
    """
    last_err = None
    for api_key in GROQ_API_KEYS:
        client = _client(api_key)
        for model in GROQ_VISION_MODELS:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content, model
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"All Groq vision models/keys failed. Last error: {last_err}")


def _extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from a model response.

    Handles all common failure modes:
      - Markdown code fences (```json ... ```)
      - Prose before/after the JSON block
      - Trailing commas before } or ]
      - Single-quoted strings
      - Truncated responses — closes open brackets/braces automatically
    """
    if not text:
        raise ValueError("Empty response from model")

    # 1. Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```', '', cleaned).strip()

    # 2. Fast path
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Find the start of the JSON object
    brace_start = cleaned.find('{')
    if brace_start == -1:
        raise ValueError(f"No JSON object found in response: {text[:300]!r}")

    candidate = cleaned[brace_start:]

    # 4. Common LLM formatting fixes
    candidate = re.sub(r',\s*([\]}])', r'\1', candidate)      # trailing commas
    candidate = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', candidate)  # single quotes

    # 5. Try as-is after fixes
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 6. Truncation repair: count open brackets/braces and close them
    #    Walk through char by char, track depth for { and [
    #    If we reach the end with unclosed scopes, append the right closers.
    in_string  = False
    escape_next = False
    stack = []   # each entry is '{' or '['
    for ch in candidate:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()

    # If we were cut off mid-string, close the string first
    if in_string:
        candidate += '"'

    # Close any open brackets/braces in reverse order
    closing = ''.join(']' if s == '[' else '}' for s in reversed(stack))
    repaired = candidate + closing

    # One more trailing-comma cleanup after repair
    repaired = re.sub(r',\s*([\]}])', r'\1', repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse JSON from model response (tried repair). "
            f"Error: {e}. First 300 chars: {text[:300]!r}"
        )


# ── Step 1: PCB Validation ───────────────────────────────────────────────────
PCB_REJECTION_CONFIDENCE_THRESHOLD = 0.85  # Only hard-reject if model is >85% sure it is NOT a PCB

_PCB_SYSTEM = (
    "You are a PCB (printed circuit board) identification expert. "
    "Your ONLY job is to determine whether the image shows a PCB. "
    "You must respond with a single valid JSON object and absolutely nothing else — "
    "no markdown, no code fences, no explanation."
)

_PCB_USER = (
    "Does this image show a PCB (printed circuit board)?\n\n"
    "PCBs include: bare boards, populated boards with components, partially assembled boards, "
    "green/blue/red/black/white FR4 substrates, flex PCBs, prototype boards. "
    "A board with soldered components IS still a PCB.\n\n"
    "ONLY set is_pcb to false if you are highly certain the image shows something completely "
    "unrelated to electronics (e.g. a face, food, document, outdoor scene).\n\n"
    "Respond with ONLY this JSON (replace values appropriately):\n"
    '{"is_pcb": true, "confidence": 0.97, "reason": "Green FR4 board with SMD components visible"}'
)


def is_pcb_image(image_path: str) -> tuple:
    """
    Returns (is_pcb: bool, confidence: float, reason: str, model_used: str).
    On any API failure, defaults to (True, 0.0, error_message, 'error') so the
    pipeline is not blocked by a transient Groq outage.
    """
    b64 = _encode_image(image_path)

    messages = [
        {"role": "system", "content": _PCB_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": _PCB_USER},
            ],
        },
    ]

    try:
        content, model_used = _try_models(messages, max_tokens=256, temperature=0.0)
        data = _extract_json(content)
        is_pcb     = bool(data.get('is_pcb', True))
        confidence = float(data.get('confidence', 0.0))
        reason     = str(data.get('reason', ''))
        # If model says NOT pcb but confidence is low → pass through anyway
        if not is_pcb and confidence < PCB_REJECTION_CONFIDENCE_THRESHOLD:
            is_pcb = True
            reason = f"[Low-confidence rejection overridden — {confidence:.0%}] {reason}"
        return (is_pcb, confidence, reason, model_used)
    except Exception as e:
        return (True, 0.0, f"Groq PCB check failed: {e}", "error")


# ── Step 2: Solder Defect Detection ─────────────────────────────────────────

_SOLDER_SYSTEM = """\
You are a certified IPC-A-610 PCB soldering quality inspector with 20 years of experience.
Your task is to examine a PCB image and report ONLY solder defects that are CLEARLY and UNAMBIGUOUSLY visible.

ABSOLUTE RULES — violating ANY of these disqualifies your response:
1. Output ONLY a single valid JSON object. No markdown, no code fences, no prose, no explanation.
2. Every string value must use double quotes. No single quotes anywhere.
3. Bounding box coordinates must be normalised floats in [0.0, 1.0] relative to the full image.
4. x_min < x_max and y_min < y_max must always hold.
5. Never report a defect you are not at least 65% certain about.
6. Never draw a bounding box larger than 30% of the image area.
7. If you see NO defects, return: {"defects": []}
"""

_SOLDER_USER = """\
Carefully inspect this PCB image for the following SOLDER defects only.
For each defect type, I describe exactly what it looks like so you do NOT hallucinate:

SOLDER_BURN
  Visual: Dark brown or black discolouration, charring, or scorched residue ON or immediately
  AROUND a solder joint or pad. The burn area looks matte-black or charcoal, clearly different
  from the silver/grey of normal solder. Often has irregular edges. May show flux residue
  turned dark amber or brown.
  HOW TO LOCALIZE: Find the darkest, most charred region. Draw your bounding box TIGHTLY
  around that specific dark patch — not around the entire component area. If the burn spans
  multiple pads, include all of them in a single box. The box should be snug.
  DO NOT flag: normal dark PCB substrate, shadow, dark component body, black IC packages,
  dark conformal coating, or any area not directly involving solder.
  EXAMPLE: If you see a dark scorched area spanning roughly the left-center of the image,
  x_min/x_max should tightly bracket that horizontal extent, not the whole board width.

SOLDER_BRIDGE
  Visual: A continuous solder blob connecting two physically separate pads or pins that should
  NOT be connected. The bridge is a visible silver/grey solder trail crossing the gap.
  DO NOT flag: intentional solder connections, wide pads that look close together.

COLD_JOINT
  Visual: A solder joint that looks dull, grainy, rough, or lumpy instead of smooth and shiny.
  The surface appears matte grey rather than bright metallic silver.
  DO NOT flag: normal aged solder that is slightly dull, dark ICs near pads.

SOLDER_BALL
  Visual: A small, isolated spherical bead of solder sitting on the PCB surface AWAY from a pad,
  clearly not attached to any pad or pin.
  DO NOT flag: flux residue bubbles, small dust particles, component leads.

EXCESS_SOLDER
  Visual: Solder that has clearly overflowed the intended pad boundaries forming large, irregular
  blobs. Much more solder than needed for the joint.
  DO NOT flag: normal rounded fillet solder joints.

INSUFFICIENT_SOLDER
  Visual: A pad where solder is barely present — you can clearly see most of the copper pad
  is exposed with only a thin film of solder, or the component lead is visibly not wetted.
  DO NOT flag: small SMD resistors/capacitors that naturally have small solder joints.

LIFTED_PAD
  Visual: A copper pad that has physically separated from the board substrate, visibly peeling
  up from the surface.
  DO NOT flag: component body edges, solder mask edges.

TOMBSTONE
  Visual: A small SMD component (resistor or capacitor) that has one end lifted off its pad
  and is standing nearly vertical.
  DO NOT flag: components that look slightly tilted due to camera angle.

IMPORTANT ANTI-HALLUCINATION RULES:
- If the board looks clean and normal, return {"defects": []}.
- Dark areas that are just shadows, component bodies, or dark PCB substrate are NOT burns.
- Only flag solder_burn if you can clearly see scorching, charring, or heat discolouration
  ON or touching a solder joint. Shadow ≠ burn.
- Only flag defects where the bounding box tightly surrounds the actual defect region.
- Boxes larger than ~15% of image area are almost certainly wrong — do not include them.
- Maximum 8 defects per image. If you think you see more, only report the 8 most certain ones.

Respond with ONLY this JSON structure (no other text whatsoever):
{"defects": [{"type": "solder_burn", "description": "Charred flux residue on R12 pad 1", "confidence": 0.83, "x_min": 0.42, "y_min": 0.31, "x_max": 0.49, "y_max": 0.38}]}
"""


def detect_solder_defects(image_path: str, img_w: int, img_h: int,
                           min_confidence: float = SOLDER_CONF_MIN) -> tuple:
    """
    Asks Groq to find solder-specific defects the YOLO model does not cover.

    Returns (defects: list, error_msg: str | None)

    Each defect dict:
      {
        'class':       'solder_burn',
        'label':       'solder_defect',
        'description': 'Charred flux near R12 pad 1',
        'confidence':  0.83,
        'severity':    0.75,
        'bbox':        (x1, y1, x2, y2),   # pixel coordinates
        'area':        <pixels²>,
        'area_pct':    0.31,               # % of total image area
        'zone':        '—',
        'source':      'groq',
      }

    Filters applied (hallucination guards):
      - confidence < min_confidence       → dropped
      - bbox area > BBOX_MAX_AREA_FRAC    → dropped (hallucinated large box)
      - bbox area < BBOX_MIN_AREA_FRAC    → dropped (sub-pixel noise)
      - bbox width or height > BBOX_MAX_SIDE_FRAC → dropped
    """
    b64 = _encode_image(image_path)

    messages = [
        {"role": "system", "content": _SOLDER_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": _SOLDER_USER},
            ],
        },
    ]

    try:
        content, model_used = _try_models(messages, max_tokens=3000, temperature=0.0)
        data = _extract_json(content)
        raw_defects = data.get('defects', [])
        if not isinstance(raw_defects, list):
            return [], "Groq returned non-list defects field"
    except Exception as e:
        return [], f"Groq detection failed: {e}"

    parsed  = []
    rejected = 0

    for d in raw_defects:
        try:
            # ── Confidence filter ────────────────────────────────────────
            conf = float(d.get('confidence', 0.0))
            if conf < min_confidence:
                rejected += 1
                continue

            stype = str(d.get('type', 'solder_defect')).lower().strip()
            # Normalise known aliases
            stype = {
                'burn':   'solder_burn',
                'bridge': 'solder_bridge',
                'ball':   'solder_ball',
                'cold':   'cold_joint',
            }.get(stype, stype)

            # ── Parse + clamp normalised → pixel coords ──────────────────
            xn1 = max(0.0, min(1.0, float(d['x_min'])))
            yn1 = max(0.0, min(1.0, float(d['y_min'])))
            xn2 = max(0.0, min(1.0, float(d['x_max'])))
            yn2 = max(0.0, min(1.0, float(d['y_max'])))

            # Ensure box is not degenerate
            if xn2 <= xn1 or yn2 <= yn1:
                rejected += 1
                continue

            box_w_frac    = xn2 - xn1
            box_h_frac    = yn2 - yn1
            box_area_frac = box_w_frac * box_h_frac

            # ── Bbox sanity / hallucination filters ──────────────────────
            if box_area_frac > BBOX_MAX_AREA_FRAC:   # box covers too much of image
                rejected += 1
                continue
            if box_area_frac < BBOX_MIN_AREA_FRAC:   # sub-pixel noise
                rejected += 1
                continue
            if box_w_frac > BBOX_MAX_SIDE_FRAC or box_h_frac > BBOX_MAX_SIDE_FRAC:
                rejected += 1
                continue

            x1   = int(xn1 * img_w)
            y1   = int(yn1 * img_h)
            x2   = int(xn2 * img_w)
            y2   = int(yn2 * img_h)
            area = (x2 - x1) * (y2 - y1)

            severity = SOLDER_SEVERITY.get(stype, SOLDER_SEVERITY['solder_defect'])

            parsed.append({
                'class':       stype,
                'label':       'solder_defect',
                'description': str(d.get('description', '')),
                'confidence':  conf,
                'severity':    severity,   # from engineering table, not LLM
                'bbox':        (x1, y1, x2, y2),
                'area':        area,
                'area_pct':    round(box_area_frac * 100, 3),
                'zone':        '—',
                'source':      'groq',
            })
        except Exception:
            rejected += 1
            continue

    return parsed, None


# ── Draw Groq detections onto an existing annotated image ────────────────────
def draw_solder_defects(img_bgr: np.ndarray, defects: list) -> np.ndarray:
    """
    Overlays Groq-detected solder defects onto a BGR image.
    Uses a distinct AMBER/GOLD colour + 'GROQ:' prefix to differentiate from YOLO boxes.
    Draws: semi-transparent filled overlay + solid border + corner brackets + label.
    This ensures the box is always clearly visible regardless of its size.
    """
    GOLD       = (0, 200, 255)     # BGR amber/gold
    GOLD_DARK  = (0, 140, 200)     # darker gold for border
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.44
    thickness  = 2

    for d in defects:
        x1, y1, x2, y2 = d['bbox']
        conf  = d['confidence']
        stype = d['class']

        # ── 1. Semi-transparent filled overlay inside the box ────────────────
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), GOLD, -1)
        cv2.addWeighted(overlay, 0.18, img_bgr, 0.82, 0, img_bgr)

        # ── 2. Solid border rectangle ─────────────────────────────────────────
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), GOLD_DARK, 2)

        # ── 3. Corner brackets on top (makes it stand out from YOLO boxes) ───
        bw = x2 - x1
        bh = y2 - y1
        seg = max(8, min(bw // 4, bh // 4, 22))
        for (sx, sy, ex, ey) in [
            (x1, y1, x1 + seg, y1), (x2 - seg, y1, x2, y1),   # top
            (x1, y2, x1 + seg, y2), (x2 - seg, y2, x2, y2),   # bottom
            (x1, y1, x1, y1 + seg), (x1, y2 - seg, x1, y2),   # left
            (x2, y1, x2, y1 + seg), (x2, y2 - seg, x2, y2),   # right
        ]:
            cv2.line(img_bgr, (sx, sy), (ex, ey), GOLD, 3)

        # ── 4. Label ──────────────────────────────────────────────────────────
        label = f"GROQ:{stype} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        lx = max(0, x1)
        ly = max(th + 6, y1 - 4)
        # Label background
        cv2.rectangle(img_bgr, (lx, ly - th - 5), (lx + tw + 6, ly + 2), GOLD, -1)
        cv2.rectangle(img_bgr, (lx, ly - th - 5), (lx + tw + 6, ly + 2), GOLD_DARK, 1)
        cv2.putText(img_bgr, label, (lx + 3, ly - 2),
                    font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    return img_bgr
