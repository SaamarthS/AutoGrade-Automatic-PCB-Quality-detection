# ═══════════════════════════════════════════════════════
# AutoGrade — PCB Quality Inspector Dashboard
# Fixed: conf threshold, label overlap, UI polish
# ═══════════════════════════════════════════════════════


import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import pandas as pd
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import HRFlowable, KeepTogether
from io import BytesIO
from grading_engine import load_model, grade_pcb, SEVERITY_WEIGHTS
import shutil
import groq_vision
from cloud_logger import log_inspection, load_log


# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="AutoGrade — PCB Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Theme state ─────────────────────────────────────────
if 'light_mode' not in st.session_state:
    st.session_state.light_mode = False

# ── CSS custom-property themes ───────────────────────────
# All colours are defined as CSS variables so a single
# class swap on <body> (data-theme) switches the whole UI.
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

  /* ── TOKEN DEFINITIONS ── */
  :root, [data-theme="dark"] {
    --bg-page:        #0d1117;
    --bg-surface:     #161b22;
    --bg-surface2:    #1c2128;
    --bg-input:       #161b22;
    --border:         #30363d;
    --text-primary:   #e6edf3;
    --text-secondary: #8b949e;
    --text-muted:     #6e7681;
    --accent:         #58a6ff;
    --accent-hover:   #79c0ff;
    --success:        #3fb950;
    --warning:        #e3b341;
    --danger:         #f85149;
    --tab-active-bg:  #0d1117;
    --sidebar-bg:     #161b22;
    --hr-color:       #30363d;
    --badge-A-bg:     #0d4429; --badge-A-fg: #3fb950; --badge-A-border: #3fb950;
    --badge-B-bg:     #2d2a00; --badge-B-fg: #e3b341; --badge-B-border: #e3b341;
    --badge-C-bg:     #2d1a00; --badge-C-fg: #f0883e; --badge-C-border: #f0883e;
    --badge-R-bg:     #3d0000; --badge-R-fg: #f85149; --badge-R-border: #f85149;
    --verdict-border: #58a6ff;
    --metric-value:   #58a6ff;
  }

  [data-theme="light"] {
    --bg-page:        #f5f7fa;
    --bg-surface:     #ffffff;
    --bg-surface2:    #eef1f6;
    --bg-input:       #ffffff;
    --border:         #d0d7de;
    --text-primary:   #1a1f2e;
    --text-secondary: #57606a;
    --text-muted:     #8c959f;
    --accent:         #0969da;
    --accent-hover:   #0860ca;
    --success:        #1a7f37;
    --warning:        #9a6700;
    --danger:         #cf222e;
    --tab-active-bg:  #f5f7fa;
    --sidebar-bg:     #ffffff;
    --hr-color:       #d0d7de;
    --badge-A-bg:     #dafbe1; --badge-A-fg: #1a7f37; --badge-A-border: #2da44e;
    --badge-B-bg:     #fff8c5; --badge-B-fg: #9a6700; --badge-B-border: #d4a72c;
    --badge-C-bg:     #fff1e5; --badge-C-fg: #bc4c00; --badge-C-border: #e16f24;
    --badge-R-bg:     #ffebe9; --badge-R-fg: #cf222e; --badge-R-border: #f85149;
    --verdict-border: #0969da;
    --metric-value:   #0969da;
  }

  /* ── BASE RESETS ── */
  html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-page) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
  }

  /* ── TYPOGRAPHY ── */
  h1 { color: var(--accent) !important; letter-spacing: -1px; font-weight: 800 !important; }
  h2, h3, h4, h5 { color: var(--text-primary) !important; font-weight: 700 !important; }
  p, li, span, label, div { font-family: 'Inter', sans-serif !important; }

  /* ── SIDEBAR ── */
  [data-testid="stSidebar"],
  [data-testid="stSidebar"] > div:first-child {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text-primary) !important; }
  [data-testid="stSidebar"] .stMarkdown h3 { color: var(--accent) !important; }
  
  [data-testid="collapsedControl"] svg {
    color: var(--text-primary) !important;
    fill: var(--text-primary) !important;
  }

  /* ── METRIC CARDS ── */
  [data-testid="metric-container"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
  }
  [data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--metric-value) !important;
    font-size: 26px !important;
    font-weight: 800 !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: var(--text-muted) !important;
  }

  /* ── GRADE BADGE ── */
  .grade-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 6px 28px;
    border-radius: 10px;
    font-size: 52px;
    font-weight: 900;
    letter-spacing: -2px;
    min-width: 90px;
    margin: 4px 0;
    border-width: 2px;
    border-style: solid;
  }
  .grade-A      { background: var(--badge-A-bg); color: var(--badge-A-fg); border-color: var(--badge-A-border); }
  .grade-B      { background: var(--badge-B-bg); color: var(--badge-B-fg); border-color: var(--badge-B-border); }
  .grade-C      { background: var(--badge-C-bg); color: var(--badge-C-fg); border-color: var(--badge-C-border); }
  .grade-Reject { background: var(--badge-R-bg); color: var(--badge-R-fg); border-color: var(--badge-R-border); }

  /* ── VERDICT BOX ── */
  .verdict-box {
    background: var(--bg-surface);
    border-left: 4px solid var(--verdict-border);
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    margin: 10px 0;
    font-size: 14.5px;
    font-weight: 500;
    color: var(--text-primary);
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }

  /* ── DIVIDER ── */
  hr { border-color: var(--hr-color) !important; }

  /* ── DATAFRAME ── */
  .stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
  }
  .stDataFrame th {
    background: var(--bg-surface2) !important;
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
  }
  .stDataFrame td { color: var(--text-primary) !important; }

  /* ── FILE UPLOADER ── */
  [data-testid="stFileUploader"],
  [data-testid="stFileUploader"] > section {
    background: var(--bg-surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 8px !important;
  }
  /* Fix inner dropzone background specifically */
  [data-testid="stFileUploadDropzone"] {
    background: var(--bg-surface2) !important;
  }
  /* Target text nodes explicitly — do NOT use * which leaks into hidden a11y spans */
  [data-testid="stFileUploader"] label,
  [data-testid="stFileUploader"] p,
  [data-testid="stFileUploader"] small,
  [data-testid="stFileUploader"] span:not([hidden]):not([aria-hidden]) {
    color: var(--text-primary) !important;
  }

  /* ── DOWNLOAD BUTTON ── */
  .stDownloadButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 10px 0 !important;
    width: 100% !important;
    transition: background 0.18s ease !important;
  }
  .stDownloadButton > button:hover { background: #2ea043 !important; }

  /* ── BUTTONS ── */
  .stButton > button {
    background: var(--bg-surface2) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
  }

  /* ── SLIDERS ── */
  [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary) !important;
  }

  /* ── TABS ── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface) !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 4px !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px 8px 0 0 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 10px 26px !important;
    border: none !important;
    transition: color 0.15s ease !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--tab-active-bg) !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-page) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 20px 18px !important;
  }

  /* ── CAMERA ── */
  [data-testid="stCameraInput"] > div {
    background: var(--bg-surface) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
  }
  [data-testid="stCameraInput"] button {
    background: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    margin-top: 8px !important;
  }

  /* ── INFO / WARNING / SUCCESS ── */
  .stAlert {
    border-radius: 10px !important;
    border-width: 1px !important;
    font-weight: 500 !important;
  }

  /* ── TOGGLE LABEL ── */
  .theme-toggle-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
  }
  .theme-toggle-label {
    font-size: 0.82rem;
    color: var(--text-secondary);
    font-weight: 600;
    letter-spacing: 0.3px;
  }

  /* ── CAPTIONS ── */
  [data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
  }

  /* ── MAIN CONTENT CONTAINER SPACING ── */
  .block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1180px !important;
  }

  /* ── FIX: Hide sidebar collapse/expand keyboard-shortcut tooltip ── */
  /* The JS below strips title attrs; this CSS hides any remaining tooltip popups */
  [data-testid="stSidebarCollapsedControl"] button,
  [data-testid="stSidebarCollapseButton"] button,
  [data-testid="stSidebarNavCollapseButton"] button {
    /* Prevent native title-attribute tooltips via CSS where possible */
  }
  /* Hide the tooltip popup that appears on hover over sidebar controls */
  [data-testid="stSidebarCollapsedControl"] button::after,
  [data-testid="stSidebarCollapseButton"] button::after {
    display: none !important;
  }
  [data-testid="stSidebarCollapsedControl"] [role="tooltip"],
  [data-testid="stSidebarCollapseButton"] [role="tooltip"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
  }

  /* ── FIX: Material icon ligature text showing as "keyboard_double_arrow_right" ── */
  /* Streamlit uses Material Symbols font ligatures for icons. When the font  */
  /* fails to load, the raw ligature name (e.g. "keyboard_double_arrow_right") */
  /* renders as visible text. Force-load the font and ensure the icons render. */
  [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', sans-serif !important;
    font-size: 24px !important;
    -webkit-font-smoothing: antialiased;
    overflow: hidden !important;
    /* Fallback: if font still doesn't load, clip the long text */
    max-width: 24px !important;
    max-height: 24px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
  }

  /* ── FIX: File uploader button — hide extra/garbled text ── */
  /* The Streamlit file uploader button shows raw icon name + "Upload" text */
  [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"],
  [data-testid="stFileUploader"] button.st-emotion-cache-abpmmt {
    font-size: 0 !important;
    overflow: hidden !important;
  }
  [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]::after,
  [data-testid="stFileUploader"] button.st-emotion-cache-abpmmt::after {
    content: "Browse files" !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
  }
  /* Hide stray span/p text inside the uploader button */
  [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] span,
  [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] p,
  [data-testid="stFileUploader"] button.st-emotion-cache-abpmmt span,
  [data-testid="stFileUploader"] button.st-emotion-cache-abpmmt p {
    font-size: 0 !important;
    line-height: 0 !important;
    overflow: hidden !important;
    width: 0 !important;
    height: 0 !important;
    display: inline !important;
  }

  /* ── FIX: Hide "Deploy" button text leak in toolbar ── */
  [data-testid="stToolbar"] [data-testid="stIconMaterial"] {
    max-width: 24px !important;
    overflow: hidden !important;
  }
</style>
""", unsafe_allow_html=True)

# ── JS Fix: Strip sidebar tooltip title attributes + floating portals ────────
# CSS cannot suppress native browser tooltips from title="..." attributes.
# This script continuously strips them and hides Streamlit tooltip portals.
st.markdown("""
<script>
(function() {
    function stripSidebarTooltips() {
        // Remove title attributes from sidebar collapse/expand buttons
        document.querySelectorAll(
            '[data-testid="stSidebarCollapsedControl"] button, ' +
            '[data-testid="stSidebarCollapseButton"] button, ' +
            '[data-testid="stSidebarNavCollapseButton"] button'
        ).forEach(function(btn) {
            if (btn.hasAttribute('title')) btn.removeAttribute('title');
        });

        // Also strip title from any parent element
        document.querySelectorAll(
            '[data-testid="stSidebarCollapsedControl"] [title], ' +
            '[data-testid="stSidebarCollapseButton"] [title]'
        ).forEach(function(el) {
            el.removeAttribute('title');
        });
    }

    // Run immediately + on every DOM mutation (Streamlit re-renders often)
    stripSidebarTooltips();
    var obs = new MutationObserver(stripSidebarTooltips);
    obs.observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ['title']});
})();
</script>
""", unsafe_allow_html=True)

# ── Apply theme: inject light-mode CSS variable overrides directly onto :root ──
# Streamlit's React architecture wipes DOM attributes on every re-render, so
# JS-based data-theme injection is unreliable. Instead we write the correct
# variable values straight into :root when light mode is active. No JS needed.
if st.session_state.light_mode:
    st.markdown("""
    <style>
      :root {
        --bg-page:        #f5f7fa !important;
        --bg-surface:     #ffffff !important;
        --bg-surface2:    #eef1f6 !important;
        --bg-input:       #ffffff !important;
        --border:         #d0d7de !important;
        --text-primary:   #1a1f2e !important;
        --text-secondary: #57606a !important;
        --text-muted:     #8c959f !important;
        --accent:         #0969da !important;
        --accent-hover:   #0860ca !important;
        --success:        #1a7f37 !important;
        --warning:        #9a6700 !important;
        --danger:         #cf222e !important;
        --tab-active-bg:  #f5f7fa !important;
        --sidebar-bg:     #ffffff !important;
        --hr-color:       #d0d7de !important;
        --badge-A-bg:     #dafbe1 !important;
        --badge-A-fg:     #1a7f37 !important;
        --badge-A-border: #2da44e !important;
        --badge-B-bg:     #fff8c5 !important;
        --badge-B-fg:     #9a6700 !important;
        --badge-B-border: #d4a72c !important;
        --badge-C-bg:     #fff1e5 !important;
        --badge-C-fg:     #bc4c00 !important;
        --badge-C-border: #e16f24 !important;
        --badge-R-bg:     #ffebe9 !important;
        --badge-R-fg:     #cf222e !important;
        --badge-R-border: #f85149 !important;
        --verdict-border: #0969da !important;
        --metric-value:   #0969da !important;
      }

      /* Streamlit's own dark-mode class overrides our :root — reset it */
      .stApp, [data-testid="stAppViewContainer"],
      [data-testid="stHeader"], [data-testid="stMain"] {
        background-color: #f5f7fa !important;
        color: #1a1f2e !important;
      }

      /* Sidebar in light mode */
      [data-testid="stSidebar"],
      [data-testid="stSidebar"] > div:first-child {
        background: #ffffff !important;
        border-right: 1px solid #d0d7de !important;
      }
      [data-testid="stSidebar"] * { color: #1a1f2e !important; }

      /* Inputs, selects, text areas */
      input, select, textarea,
      [data-baseweb="input"] input,
      [data-baseweb="select"] [data-testid="stSelectbox"] {
        background: #ffffff !important;
        color: #1a1f2e !important;
        border-color: #d0d7de !important;
      }

      /* Tab panel in light */
      .stTabs [data-baseweb="tab-panel"] {
        background: #f5f7fa !important;
      }

      /* Markdown and general text */
      .stMarkdown, .stMarkdown p, .stMarkdown li,
      .stMarkdown span, .element-container p {
        color: #1a1f2e !important;
      }

      /* Dataframe and standard tables */
      [data-testid="stDataFrame"] td, 
      [data-testid="stDataFrame"] th,
      [data-testid="stTable"] td,
      [data-testid="stTable"] th {
        color: #1a1f2e !important; 
        background: #ffffff !important;
        border-color: #d0d7de !important;
      }
      [data-testid="stDataFrame"] tr:nth-child(even) td,
      [data-testid="stTable"] tr:nth-child(even) td { 
        background: #f5f7fa !important; 
      }
      
      /* File uploader button fix for light mode */
      [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]::after,
      [data-testid="stFileUploader"] button.st-emotion-cache-abpmmt::after {
        color: #1a1f2e !important;
        background-color: #f6f8fa !important;
        border: 1px solid #d0d7de !important;
        padding: 4px 12px !important;
        border-radius: 6px !important;
        display: inline-block !important;
      }
      [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover::after,
      [data-testid="stFileUploader"] button.st-emotion-cache-abpmmt:hover::after {
        background-color: #f3f4f6 !important;
        border-color: #1a1f2e !important;
      }
      
      /* General buttons in light mode */
      button[data-testid="baseButton-secondary"] {
        color: #1a1f2e !important;
        background: #ffffff !important;
        border-color: #d0d7de !important;
      }
    </style>
    """, unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model('best.pt')

model = get_model()

# ── Annotated image with clean labels ───────────────────
def get_annotated_image_clean(image_path, model, conf=0.45, iou=0.4):
    """
    Runs detection and draws clean, non-overlapping labels.
    """
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        verbose=False
    )
    result = results[0]

    img = cv2.imread(image_path)
    if img is None:
        img = np.array(Image.open(image_path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    CLASS_COLORS = {
        'open_circuit':    (255, 50,  50),
        'short':           (50,  50,  255),
        'mouse_bite':      (50,  200, 50),
        'spur':            (255, 165, 0),
        'spurious_copper': (180, 0,   255),
        'missing_hole':    (0,   200, 220),
        # DeepPCB names
        'open':            (255, 50,  50),
        'mousebite':       (50,  200, 50),
        'copper':          (180, 0,   255),
        'pin-hole':        (0,   200, 220),
    }

    CLASSES = result.names  # use model's class names

    CLASS_CONF_THRESHOLD = {
        'missing_hole':    0.65,
        'open_circuit':    0.40,
        'short':           0.40,
        'mouse_bite':      0.45,
        'spur':             0.40,
        'spurious_copper': 0.40,
    }

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASSES[cls_id]
        conf_val = float(box.conf[0])

        min_conf = CLASS_CONF_THRESHOLD.get(cls_name, 0.45)
        if conf_val < min_conf:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = CLASS_COLORS.get(cls_name, (100, 100, 255))

        # Draw box (thicker = more visible)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background ABOVE the box (not on top of it)
        label = f"{cls_name} {conf_val:.2f}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness  = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Position label above box, clipped to image boundary
        label_x = max(0, x1)
        label_y = max(th + 4, y1 - 4)

        # Background rectangle for label
        cv2.rectangle(
            img,
            (label_x, label_y - th - 4),
            (label_x + tw + 4, label_y + 2),
            color, -1
        )

        # White text
        cv2.putText(
            img, label,
            (label_x + 2, label_y - 2),
            font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Preprocessing pipeline ──────────────────────────────
def preprocess_pcb(image_path, use_clahe=True, clahe_clip=2.0, clahe_grid=8,
                   use_sharpen=True, sharpen_strength=1.5,
                   use_denoise=True, denoise_h=6):
    """
    Applies a preprocessing pipeline to a PCB image to improve model detection
    on real-world photos (bridging the domain gap from training data).

    Steps (each individually toggleable):
      1. CLAHE  — boosts local contrast in LAB colour space (reveals defects)
      2. Unsharp mask  — sharpens edges (open circuits, spurs become crisper)
      3. Fast NL-means denoise  — removes camera sensor noise / JPEG artefacts

    Returns: preprocessed BGR numpy array + RGB version for display.
    """
    img = cv2.imread(image_path)
    if img is None:
        img = np.array(Image.open(image_path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Step 1 — CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Applied to the L channel in LAB space so colours are preserved.
    if use_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                 tileGridSize=(clahe_grid, clahe_grid))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Step 2 — Unsharp masking (sharpens fine defect edges)
    if use_sharpen:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
        img = cv2.addWeighted(img, 1 + sharpen_strength,
                              blurred, -sharpen_strength, 0)

    # Step 3 — Fast NL-means denoising (reduces graininess / JPEG compression)
    if use_denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None,
                                               h=denoise_h, hColor=denoise_h,
                                               templateWindowSize=7,
                                               searchWindowSize=21)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, rgb   # (BGR for saving, RGB for display)


# ── PDF Generator (ReportLab) ─────────────────────────────────────────────────
# Premium A4 report with bleed header, structured grade card, two-column images,
# an enhanced defect table (with Source column), and a multi-line footer.
# ──────────────────────────────────────────────────────────────────────────────
def generate_pdf(result, original_path, annotated_path):
    buf  = BytesIO()
    IST  = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    now  = datetime.datetime.now(IST)
    grade = result['grade']
    score = result['score']

    # ── Page geometry ────────────────────────────────────────────────────────
    L_MAR, R_MAR = 18*mm, 18*mm
    T_MAR, B_MAR =  0*mm, 16*mm
    PAGE_W = A4[0]
    PAGE_H = A4[1]
    W = PAGE_W - L_MAR - R_MAR        # usable body width
    BLEED_W = PAGE_W                   # full page width for header/footer

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=L_MAR, rightMargin=R_MAR,
        topMargin=T_MAR,  bottomMargin=B_MAR,
        title='AutoGrade — PCB Quality Inspection Report',
        author='AutoGrade, RVCE EL Project',
    )

    # ── Colour palette ───────────────────────────────────────────────────────
    C = lambda h: colors.HexColor(h)
    NAVY         = C('#0f1f3d')
    NAVY_MID     = C('#1a3560')
    ACCENT_BLUE  = C('#1e5aa8')
    ACCENT_LIGHT = C('#e8f0fb')
    WHITE        = colors.white
    OFF_WHITE    = C('#f8f9fc')
    LIGHT_GREY   = C('#f2f4f8')
    MID_GREY     = C('#d8dde8')
    DARK_GREY    = C('#4a5568')
    BLACK        = C('#1a202c')

    GRADE_FG = {'A': C('#1a7f37'), 'B': C('#7d5900'), 'C': C('#bc4c00'), 'Reject': C('#b31d1d')}
    GRADE_BG = {'A': C('#dafbe1'), 'B': C('#fff8c5'), 'C': C('#fff1e5'), 'Reject': C('#ffebe9')}
    GRADE_BD = {'A': C('#2da44e'), 'B': C('#d4a72c'), 'C': C('#e16f24'), 'Reject': C('#f85149')}

    g_fg = GRADE_FG.get(grade, DARK_GREY)
    g_bg = GRADE_BG.get(grade, LIGHT_GREY)
    g_bd = GRADE_BD.get(grade, MID_GREY)

    # ── Style factory ────────────────────────────────────────────────────────
    _uid = [0]
    def PS(base='', **kw):
        _uid[0] += 1
        return ParagraphStyle(f'_s{_uid[0]}_{base}', **kw)

    # base styles
    s_body   = PS('body',  fontSize=8.5, textColor=DARK_GREY,  fontName='Helvetica',      leading=12)
    s_small  = PS('small', fontSize=7.5, textColor=DARK_GREY,  fontName='Helvetica',      leading=10)
    s_foot   = PS('foot',  fontSize=7,   textColor=DARK_GREY,  fontName='Helvetica',      alignment=TA_CENTER, leading=10)
    s_cap    = PS('cap',   fontSize=7.5, textColor=DARK_GREY,  fontName='Helvetica-Oblique', alignment=TA_CENTER, leading=10, spaceBefore=3)

    # header
    s_hdr_title = PS('ht', fontSize=22, textColor=WHITE,         fontName='Helvetica-Bold', alignment=TA_CENTER, leading=28)
    s_hdr_sub   = PS('hs', fontSize=9,  textColor=C('#b0c4e8'), fontName='Helvetica',      alignment=TA_CENTER, leading=13)

    # section heading
    s_sec = PS('sec', fontSize=9.5, textColor=NAVY, fontName='Helvetica-Bold',
               spaceBefore=0, spaceAfter=2, leading=13, borderPadding=(0,0,2,0))

    # table header / cells
    s_th   = PS('th',  fontSize=8,   textColor=WHITE,     fontName='Helvetica-Bold', alignment=TA_CENTER, leading=11)
    s_th_l = PS('thl', fontSize=8,   textColor=WHITE,     fontName='Helvetica-Bold', alignment=TA_LEFT,   leading=11)
    s_td   = PS('td',  fontSize=8,   textColor=BLACK,     fontName='Helvetica',      alignment=TA_CENTER, leading=11)
    s_td_l = PS('tdl', fontSize=8,   textColor=BLACK,     fontName='Helvetica',      alignment=TA_LEFT,   leading=11)
    s_td_r = PS('tdr', fontSize=8,   textColor=DARK_GREY, fontName='Helvetica',      alignment=TA_RIGHT,  leading=11)
    s_mono = PS('mon', fontSize=7.5, textColor=BLACK,     fontName='Courier',        alignment=TA_CENTER, leading=11)

    # grade card
    s_grade_ltr  = PS('gl',  fontSize=44, textColor=g_fg,     fontName='Helvetica-Bold', alignment=TA_CENTER, leading=50)
    s_grade_lbl  = PS('glb', fontSize=7,  textColor=g_fg,     fontName='Helvetica-Bold', alignment=TA_CENTER, leading=10)
    s_score_big  = PS('sb',  fontSize=30, textColor=BLACK,    fontName='Helvetica-Bold', alignment=TA_CENTER, leading=36)
    s_score_lbl  = PS('sbl', fontSize=7,  textColor=DARK_GREY,fontName='Helvetica',      alignment=TA_CENTER, leading=10)
    s_stat_val   = PS('stv', fontSize=18, textColor=BLACK,    fontName='Helvetica-Bold', alignment=TA_CENTER, leading=22)
    s_stat_lbl   = PS('stl', fontSize=7,  textColor=DARK_GREY,fontName='Helvetica',      alignment=TA_CENTER, leading=10)
    s_action     = PS('act', fontSize=8,  textColor=DARK_GREY,fontName='Helvetica-Oblique', alignment=TA_CENTER, leading=11)

    story = []

    # ══════════════════════════════════════════════════════════════════════════
    # 1. FULL-BLEED HEADER BANNER
    # The table is widened to BLEED_W and positioned using negative insets via
    # a wrapper Table that spans the full page width including margins.
    # ══════════════════════════════════════════════════════════════════════════
    hdr_inner = Table(
        [[Paragraph('AutoGrade', s_hdr_title)],
         [Paragraph('PCB Quality Inspection Report', s_hdr_sub)]],
        colWidths=[W]
    )
    hdr_inner.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), NAVY),
        ('TOPPADDING',    (0,0), (0,0),   16),
        ('BOTTOMPADDING', (0,0), (0,0),    4),
        ('TOPPADDING',    (0,1), (0,1),    0),
        ('BOTTOMPADDING', (0,1), (0,1),   14),
        ('LEFTPADDING',   (0,0), (-1,-1),  0),
        ('RIGHTPADDING',  (0,0), (-1,-1),  0),
    ]))
    # Bleed wrapper: extend left/right by the margin amounts so it touches edges
    bleed_hdr = Table(
        [[Spacer(L_MAR, 1), hdr_inner, Spacer(R_MAR, 1)]],
        colWidths=[L_MAR, W, R_MAR]
    )
    bleed_hdr.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), NAVY),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING',(0,0), (-1,-1), 0),
        ('TOPPADDING',  (0,0), (-1,-1), 0),
        ('BOTTOMPADDING',(0,0),(-1,-1), 0),
    ]))
    story.append(bleed_hdr)
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════════════════════
    # 2. REPORT META INFO BAR  (timestamp · filename · model)
    # ══════════════════════════════════════════════════════════════════════════
    ts_str   = now.strftime('%d %B %Y   %I:%M %p')
    src_name = os.path.basename(original_path) if original_path else 'N/A'
    meta_tbl = Table(
        [[
            Paragraph(f'<b>Generated:</b>  {ts_str}', s_small),
            Paragraph(f'<b>File:</b>  {src_name}',    PS('fn', fontSize=7.5, textColor=DARK_GREY,
                                                          fontName='Helvetica', alignment=TA_CENTER, leading=10)),
            Paragraph('<b>System:</b>  AutoGrade v1.0', PS('sv', fontSize=7.5, textColor=DARK_GREY,
                                                            fontName='Helvetica', alignment=TA_RIGHT, leading=10)),
        ]],
        colWidths=[W*0.45, W*0.28, W*0.27]
    )
    meta_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), LIGHT_GREY),
        ('BOX',           (0,0), (-1,-1), 0.5, MID_GREY),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('RIGHTPADDING',  (0,0), (-1,-1), 8),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. GRADE CARD
    # Left: grade letter (colour-coded).  Right: 3 stat boxes side-by-side.
    # Bottom: recommended action text.
    # ══════════════════════════════════════════════════════════════════════════
    action_map = {
        'A':      'Approve for shipping — meets all quality standards.',
        'B':      'Minor rework recommended — re-inspect after correction.',
        'C':      'Mandatory rework required before use.',
        'Reject': 'Critical defects detected — scrap or complete rework.',
    }

    # Grade letter cell
    grade_cell = Table(
        [[Paragraph(grade, s_grade_ltr)],
         [Paragraph('GRADE', s_grade_lbl)]],
        colWidths=[W * 0.20]
    )
    grade_cell.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), g_bg),
        ('TOPPADDING',    (0,0), (0,0),   12),
        ('BOTTOMPADDING', (0,0), (0,0),    2),
        ('TOPPADDING',    (0,1), (0,1),    0),
        ('BOTTOMPADDING', (0,1), (0,1),   10),
        ('LEFTPADDING',   (0,0), (-1,-1),  4),
        ('RIGHTPADDING',  (0,0), (-1,-1),  4),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('LINEAFTER',     (0,0), (0,-1),  1.5, g_bd),
    ]))

    # Score cell
    score_cell = Table(
        [[Paragraph(str(score), s_score_big)],
         [Paragraph('/100  Quality Score', s_score_lbl)]],
        colWidths=[W * 0.25]
    )
    score_cell.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), OFF_WHITE),
        ('TOPPADDING',    (0,0), (0,0),   10),
        ('BOTTOMPADDING', (0,0), (0,0),    2),
        ('TOPPADDING',    (0,1), (0,1),    0),
        ('BOTTOMPADDING', (0,1), (0,1),   10),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('LINEAFTER',     (0,0), (0,-1),  0.5, MID_GREY),
    ]))

    # Defect count cell
    def _stat_cell(val, label, width):
        t = Table(
            [[Paragraph(str(val), s_stat_val)],
             [Paragraph(label,    s_stat_lbl)]],
            colWidths=[width]
        )
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), OFF_WHITE),
            ('TOPPADDING',    (0,0), (0,0),   10),
            ('BOTTOMPADDING', (0,0), (0,0),    2),
            ('TOPPADDING',    (0,1), (0,1),    0),
            ('BOTTOMPADDING', (0,1), (0,1),   10),
            ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
            ('LINEAFTER',     (0,0), (0,-1),  0.5, MID_GREY),
        ]))
        return t

    stat_w = (W - W*0.20 - W*0.25) / 3   # remaining width split evenly into 3 stat cells
    card_row = Table(
        [[grade_cell, score_cell,
          _stat_cell(result['defect_count'], 'Defects Found', stat_w),
          _stat_cell(f"{result['cleanliness']}%", 'Cleanliness', stat_w),
          _stat_cell(result['D'], 'Severity (D)', stat_w)]],
        colWidths=[W*0.20, W*0.25, stat_w, stat_w, stat_w]
    )
    card_row.setStyle(TableStyle([
        ('BOX',           (0,0), (-1,-1), 1.0, g_bd),
        ('TOPPADDING',    (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING',   (0,0), (-1,-1), 0),
        ('RIGHTPADDING',  (0,0), (-1,-1), 0),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(KeepTogether([
        card_row,
        Spacer(1, 2),
        Table(
            [[Paragraph(f'Recommended Action:  {action_map.get(grade, "")}', s_action)]],
            colWidths=[W]
        ),
    ]))
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. SCORE BREAKDOWN TABLE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width='100%', thickness=0.5, color=MID_GREY, spaceAfter=3))
    story.append(Paragraph('Score Breakdown', s_sec))
    story.append(HRFlowable(width='100%', thickness=0.5, color=MID_GREY, spaceAfter=4))

    # Compute colour for each score bar (used in value column)
    breakdown_rows = [
        [Paragraph(h, s_th) for h in ['Metric', 'Symbol', 'Value', 'Weight', 'Description']],
        [
            Paragraph('Defect Severity', s_td_l),
            Paragraph('D',   s_mono),
            Paragraph(str(result['D']),   s_td),
            Paragraph('40 %', s_td),
            Paragraph('Weighted sum: severity × confidence × area', s_td_l),
        ],
        [
            Paragraph('Defect Density', s_td_l),
            Paragraph('\u03c1',  s_mono),          # rho
            Paragraph(str(result['rho']), s_td),
            Paragraph('30 %', s_td),
            Paragraph('Defects per 10,000 px\u00b2 of board area', s_td_l),
        ],
        [
            Paragraph('Zone Penalty', s_td_l),
            Paragraph('Z',   s_mono),
            Paragraph(str(result['Z']),   s_td),
            Paragraph('30 %', s_td),
            Paragraph('1.5\u00d7 multiplier for centre-zone defects', s_td_l),
        ],
        [
            Paragraph('<b>Final Score</b>', PS('fs', fontSize=8, textColor=NAVY,
                                              fontName='Helvetica-Bold', alignment=TA_LEFT, leading=11)),
            Paragraph('', s_td),
            Paragraph(f'<b>{score}</b>', PS('fsc', fontSize=9, textColor=g_fg,
                                            fontName='Helvetica-Bold', alignment=TA_CENTER, leading=12)),
            Paragraph('', s_td),
            Paragraph('100 \u2212 (0.40\u00d7D + 0.30\u00d7\u03c1 + 0.30\u00d7Z) \u00d7 100',
                      PS('fo', fontSize=7.5, textColor=DARK_GREY, fontName='Courier',
                         alignment=TA_LEFT, leading=10)),
        ],
    ]
    col_w = [W*0.26, W*0.09, W*0.10, W*0.10, 0]
    col_w[-1] = W - sum(col_w[:-1])   # fill remaining space exactly
    bt = Table(breakdown_rows, colWidths=col_w, repeatRows=1)
    bt.setStyle(TableStyle([
        # header row
        ('BACKGROUND',    (0,0), (-1,0),  ACCENT_BLUE),
        ('TOPPADDING',    (0,0), (-1,0),  6),
        ('BOTTOMPADDING', (0,0), (-1,0),  6),
        # alternating rows
        ('ROWBACKGROUNDS',(0,1), (-1,-2), [WHITE, LIGHT_GREY]),
        # summary/total row
        ('BACKGROUND',    (0,-1), (-1,-1), ACCENT_LIGHT),
        ('TOPPADDING',    (0,-1), (-1,-1), 6),
        ('BOTTOMPADDING', (0,-1), (-1,-1), 6),
        # borders
        ('BOX',           (0,0), (-1,-1), 0.5, MID_GREY),
        ('INNERGRID',     (0,0), (-1,-1), 0.3, MID_GREY),
        # padding
        ('TOPPADDING',    (0,1), (-1,-2), 5),
        ('BOTTOMPADDING', (0,1), (-1,-2), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 7),
        ('RIGHTPADDING',  (0,0), (-1,-1), 7),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(bt)
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════════════════════
    # 5. INSPECTION IMAGES  (two equal columns, with border + caption)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width='100%', thickness=0.5, color=MID_GREY, spaceAfter=3))
    story.append(Paragraph('Inspection Images', s_sec))
    story.append(HRFlowable(width='100%', thickness=0.5, color=MID_GREY, spaceAfter=4))

    GAP   = 5*mm
    img_w = (W - GAP) / 2
    img_h = img_w * 0.72           # fixed aspect ratio — proportional will shrink-to-fit

    try:
        orig_img = RLImage(original_path,  width=img_w, height=img_h, kind='proportional')
        ann_img  = RLImage(annotated_path, width=img_w, height=img_h, kind='proportional')

        def _img_box(img_el, caption_txt):
            """Wrap an image in a bordered cell with a caption row below."""
            t = Table(
                [[img_el],
                 [Paragraph(caption_txt, s_cap)]],
                colWidths=[img_w]
            )
            t.setStyle(TableStyle([
                ('BOX',           (0,0), (-1,-1), 0.5, MID_GREY),
                ('BACKGROUND',    (0,0), (0,0),   OFF_WHITE),
                ('BACKGROUND',    (0,1), (0,1),   LIGHT_GREY),
                ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
                ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING',    (0,0), (0,0),   4),
                ('BOTTOMPADDING', (0,0), (0,0),   4),
                ('TOPPADDING',    (0,1), (0,1),   3),
                ('BOTTOMPADDING', (0,1), (0,1),   4),
                ('LEFTPADDING',   (0,0), (-1,-1), 0),
                ('RIGHTPADDING',  (0,0), (-1,-1), 0),
            ]))
            return t

        # Build two image boxes and lay them out with a gap column directly
        orig_box = _img_box(orig_img, 'Original Image')
        ann_box  = _img_box(ann_img,  'Annotated \u2014 YOLO (blue) + Groq (gold)')
        img_with_gap = Table(
            [[orig_box, Spacer(GAP, 1), ann_box]],
            colWidths=[img_w, GAP, img_w]
        )
        img_with_gap.setStyle(TableStyle([
            ('LEFTPADDING',   (0,0), (-1,-1), 0),
            ('RIGHTPADDING',  (0,0), (-1,-1), 0),
            ('TOPPADDING',    (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(img_with_gap)
    except Exception as _img_err:
        story.append(Paragraph(f'(Images unavailable: {_img_err})', s_body))

    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════════════════════
    # 6. DEFECT TABLE  (# | Class | Source | Confidence | Severity | Zone)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width='100%', thickness=0.5, color=MID_GREY, spaceAfter=3))
    story.append(Paragraph('Detected Defects', s_sec))
    story.append(HRFlowable(width='100%', thickness=0.5, color=MID_GREY, spaceAfter=4))

    SOLDER_SEVERITY_PDF = {
        'solder_bridge': 0.88, 'lifted_pad': 0.82, 'tombstone': 0.80,
        'solder_burn': 0.75, 'cold_joint': 0.70, 'solder_ball': 0.65,
        'solder_defect': 0.65, 'insufficient_solder': 0.60, 'excess_solder': 0.50,
    }

    hdr_cols = ['#', 'Defect Class', 'Source', 'Confidence', 'Severity', 'Zone']
    defect_rows = [[Paragraph(h, s_th if i > 0 else s_th) for i, h in enumerate(hdr_cols)]]
    defect_rows[0][0] = Paragraph('#', s_th)
    defect_rows[0][1] = Paragraph('Defect Class', s_th_l)

    if result['defects']:
        for i, d in enumerate(result['defects'], 1):
            src     = d.get('source', 'yolo')
            src_lbl = 'Groq' if src == 'groq' else 'YOLO'
            sev     = SEVERITY_WEIGHTS.get(d['class'],
                        SOLDER_SEVERITY_PDF.get(d['class'], 0.50))
            sev_str = f"{sev:.2f}"
            conf_str = f"{d['confidence']:.3f}"
            zone_str = str(d.get('zone', '—'))
            defect_rows.append([
                Paragraph(str(i),          s_td),
                Paragraph(d['class'],       s_td_l),
                Paragraph(src_lbl,          s_td),
                Paragraph(conf_str,         s_mono),
                Paragraph(sev_str,          s_td),
                Paragraph(zone_str,         s_td),
            ])
    else:
        defect_rows.append([
            Paragraph('—', s_td),
            Paragraph('No defects detected — board is clean.', s_td_l),
            Paragraph('', s_td), Paragraph('', s_td),
            Paragraph('', s_td), Paragraph('', s_td),
        ])

    d_col_w = [W*0.055, W*0.30, W*0.09, W*0.12, W*0.11, W*0.125]
    # adjust last col to fill remaining space
    d_col_w[-1] = W - sum(d_col_w[:-1])
    dt = Table(defect_rows, colWidths=d_col_w, repeatRows=1)
    dt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  ACCENT_BLUE),
        ('TOPPADDING',    (0,0), (-1,0),  6),
        ('BOTTOMPADDING', (0,0), (-1,0),  6),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GREY]),
        ('BOX',           (0,0), (-1,-1), 0.5, MID_GREY),
        ('INNERGRID',     (0,0), (-1,-1), 0.3, MID_GREY),
        ('TOPPADDING',    (0,1), (-1,-1), 5),
        ('BOTTOMPADDING', (0,1), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 6),
        ('RIGHTPADDING',  (0,0), (-1,-1), 6),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(dt)

    # ══════════════════════════════════════════════════════════════════════════
    # 7. GRADE LEGEND  (compact reference block)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 4*mm))
    legend_data = [
        [Paragraph('Grade', s_th),
         Paragraph('Score Range', s_th),
         Paragraph('Meaning', s_th_l),
         Paragraph('Action', s_th_l)],
        [Paragraph('A', PS('lA', fontSize=8, textColor=C('#1a7f37'), fontName='Helvetica-Bold', alignment=TA_CENTER)),
         Paragraph('90 – 100', s_td), Paragraph('Excellent quality', s_td_l), Paragraph('Approve for shipping', s_td_l)],
        [Paragraph('B', PS('lB', fontSize=8, textColor=C('#7d5900'), fontName='Helvetica-Bold', alignment=TA_CENTER)),
         Paragraph('75 – 89',  s_td), Paragraph('Good, minor issues', s_td_l), Paragraph('Minor rework, re-inspect', s_td_l)],
        [Paragraph('C', PS('lC', fontSize=8, textColor=C('#bc4c00'), fontName='Helvetica-Bold', alignment=TA_CENTER)),
         Paragraph('60 – 74',  s_td), Paragraph('Rework required', s_td_l),    Paragraph('Mandatory rework before use', s_td_l)],
        [Paragraph('Reject', PS('lR', fontSize=8, textColor=C('#b31d1d'), fontName='Helvetica-Bold', alignment=TA_CENTER)),
         Paragraph('< 60',     s_td), Paragraph('Critical defects', s_td_l),   Paragraph('Scrap or complete rework', s_td_l)],
    ]
    lg = Table(legend_data, colWidths=[W*0.10, W*0.18, W*0.35, W*0.37], repeatRows=1)
    lg.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  NAVY_MID),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GREY]),
        ('BOX',           (0,0), (-1,-1), 0.5, MID_GREY),
        ('INNERGRID',     (0,0), (-1,-1), 0.3, MID_GREY),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 6),
        ('RIGHTPADDING',  (0,0), (-1,-1), 6),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(lg)

    # ══════════════════════════════════════════════════════════════════════════
    # 8. FOOTER
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width='100%', thickness=0.8, color=MID_GREY, spaceAfter=4))
    footer_tbl = Table(
        [[
            Paragraph('AutoGrade  \u2022  RV College of Engineering  \u2022  CSE Dept', s_foot),
            Paragraph(f'Team 110  \u2022  EL Project 2025-26', PS('fr', fontSize=7,
                textColor=DARK_GREY, fontName='Helvetica', alignment=TA_RIGHT, leading=10)),
        ]],
        colWidths=[W*0.60, W*0.40]
    )
    footer_tbl.setStyle(TableStyle([
        ('TOPPADDING',    (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING',   (0,0), (-1,-1), 0),
        ('RIGHTPADDING',  (0,0), (-1,-1), 0),
        ('VALIGN',        (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(footer_tbl)

    doc.build(story)
    return buf.getvalue()

# ═══════════════════════════════════════════════════════
# UI LAYOUT
# ═══════════════════════════════════════════════════════

# ── Header row  (title left · theme toggle right) ────────────
_hcol_l, _hcol_r = st.columns([5, 1])
with _hcol_l:
    st.markdown("""
    <h1 style='font-size:2.4rem; margin-bottom:0; margin-top:0'>🔬 AutoGrade</h1>
    <p style='color:var(--text-secondary); font-size:0.95rem; margin-top:2px; font-weight:500'>
        PCB Surface Defect Detection &amp; Multi-Parameter Quality Grading
        &nbsp;·&nbsp;
        <span style='color:var(--success); font-weight:600'>RVCE EL Project 2025-26</span>
    </p>
    """, unsafe_allow_html=True)
with _hcol_r:
    st.markdown('<div style="padding-top:18px; text-align:right">', unsafe_allow_html=True)
    _lm_new = st.toggle(
        "☀️ Light mode" if not st.session_state.light_mode else "🌙 Dark mode",
        value=st.session_state.light_mode,
        key='_theme_toggle',
        help="Switch between dark and light UI theme"
    )
    if _lm_new != st.session_state.light_mode:
        st.session_state.light_mode = _lm_new
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ── Sidebar — Settings ──────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.20, max_value=0.80,
        value=0.45, step=0.05,
        help="Higher = fewer but more confident detections. Recommended: 0.45"
    )
    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.20, max_value=0.70,
        value=0.40, step=0.05,
        help="Lower = more aggressive duplicate box removal."
    )
    st.divider()

    # ── Preprocessing controls ──────────────────────────
    st.markdown("### 🖼️ Image Preprocessing")
    st.caption("Helps real-world photos look more like training data.")

    use_preprocessing = st.toggle("Enable Preprocessing", value=False,
                                   help="Applies CLAHE, sharpening and denoising before detection.")

    if use_preprocessing:
        st.markdown("**CLAHE** — local contrast boost")
        use_clahe     = st.checkbox("Apply CLAHE", value=True)
        clahe_clip    = st.slider("Clip Limit",   1.0, 6.0, 2.0, 0.5,
                                   help="Higher = stronger contrast. 2–3 is usually best.")
        clahe_grid    = st.select_slider("Tile Grid Size", [4, 8, 16], value=8,
                                          help="Smaller tiles = more localised contrast.")

        st.markdown("**Unsharp Mask** — edge sharpening")
        use_sharpen       = st.checkbox("Apply Sharpening", value=True)
        sharpen_strength  = st.slider("Sharpen Strength", 0.5, 3.0, 1.5, 0.25,
                                       help="1.5 is a good starting point.")

        st.markdown("**Denoise** — remove sensor/JPEG noise")
        use_denoise   = st.checkbox("Apply Denoising", value=True)
        denoise_h     = st.slider("Denoise Strength", 3, 15, 6, 1,
                                   help="Higher = smoother. Don't go above 10 or you'll lose real detail.")
    else:
        use_clahe = use_sharpen = use_denoise = False
        clahe_clip = 2.0; clahe_grid = 8
        sharpen_strength = 1.5; denoise_h = 6

    st.divider()
    st.markdown("### 📊 Grading Weights")
    st.markdown("""
    | Component | Weight |
    |---|---|
    | Defect Severity | 40% |
    | Defect Density  | 30% |
    | Zone Penalty    | 30% |
    """)
    st.divider()
    st.markdown("### 🎯 Grade Thresholds")
    st.markdown("""
    | Grade | Score |
    |---|---|
    | 🟢 A — Excellent | 90–100 |
    | 🟡 B — Good      | 75–89  |
    | 🟠 C — Rework    | 60–74  |
    | 🔴 Reject        | < 60   |
    """)

# ── Input Tabs ──────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["📁  Upload Image", "📷  Capture PCB"])

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'camera_key' not in st.session_state:
    st.session_state.camera_key = 0

image_source = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload PCB Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, PNG",
        key=f"upload_{st.session_state.uploader_key}"
    )
    if uploaded_file:
        image_source = uploaded_file

with tab_camera:
    st.markdown("""
    <p style='color:var(--text-secondary); font-size:0.88rem; margin:0 0 6px 0'>
        📐 Align your PCB within the <span style='color:var(--accent)'>blue guide rectangle</span>,
        then click <strong style='color:var(--text-primary)'>Take Photo</strong>.
    </p>
    """, unsafe_allow_html=True)

    camera_image = st.camera_input("", label_visibility="collapsed", key=f"camera_{st.session_state.camera_key}")

    # ── Rectangle overlay on live webcam feed ──────────────
    st.markdown("""
    <style>
    @keyframes pulse-border {
      0% { box-shadow: 0 0 0 9999px rgba(0,0,0,0.40), 0 0 0 0 rgba(88,166,255,0.7); }
      50% { box-shadow: 0 0 0 9999px rgba(0,0,0,0.40), 0 0 0 10px rgba(88,166,255,0.0); }
      100% { box-shadow: 0 0 0 9999px rgba(0,0,0,0.40), 0 0 0 0 rgba(88,166,255,0); }
    }
    @keyframes pulse-text {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    </style>
    <script>
    (function() {
        function injectOverlay() {
            var videoEl = document.querySelector('[data-testid="stCameraInput"] video');
            if (!videoEl) { setTimeout(injectOverlay, 300); return; }
            if (document.getElementById('ag-pcb-overlay')) return;

            var parent = videoEl.parentElement;
            var cs = window.getComputedStyle(parent);
            if (cs.position === 'static') parent.style.position = 'relative';

            var ov = document.createElement('div');
            ov.id = 'ag-pcb-overlay';
            ov.style.cssText = [
                'position:absolute','top:0','left:0','right:0','bottom:40px',
                'pointer-events:none','z-index:20','display:flex',
                'align-items:center','justify-content:center'
            ].join(';');

            ov.innerHTML = [
                '<div style="position:relative;width:72%;height:62%;',
                'border:2px dashed rgba(88,166,255,0.6);border-radius:8px;',
                'animation: pulse-border 2s infinite;">',
                  /* label */
                  '<div style="position:absolute;top:-28px;left:50%;',
                  'transform:translateX(-50%);color:#58a6ff;font-size:12px;',
                  'font-weight:bold;letter-spacing:3px;white-space:nowrap;',
                  'text-shadow:0 0 8px rgba(88,166,255,0.8); animation: pulse-text 2s infinite;">',
                  'HOLD STEADY ALIGN PCB</div>',
                  /* corners */
                  '<div style="position:absolute;top:-4px;left:-4px;width:24px;height:24px;border-top:4px solid #58a6ff;border-left:4px solid #58a6ff;border-radius:8px 0 0 0;"></div>',
                  '<div style="position:absolute;top:-4px;right:-4px;width:24px;height:24px;border-top:4px solid #58a6ff;border-right:4px solid #58a6ff;border-radius:0 8px 0 0;"></div>',
                  '<div style="position:absolute;bottom:-4px;left:-4px;width:24px;height:24px;border-bottom:4px solid #58a6ff;border-left:4px solid #58a6ff;border-radius:0 0 0 8px;"></div>',
                  '<div style="position:absolute;bottom:-4px;right:-4px;width:24px;height:24px;border-bottom:4px solid #58a6ff;border-right:4px solid #58a6ff;border-radius:0 0 8px 0;"></div>',
                  /* crosshairs */
                  '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:24px;height:2px;background:rgba(88,166,255,0.6)"></div>',
                  '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:2px;height:24px;background:rgba(88,166,255,0.6)"></div>',
                '</div>'
            ].join('');

            parent.appendChild(ov);
        }
        var obs = new MutationObserver(function() {
            if (!document.getElementById('ag-pcb-overlay')) injectOverlay();
        });
        obs.observe(document.body, {childList:true, subtree:true});
        injectOverlay();
    })();
    </script>
    """, unsafe_allow_html=True)

    if camera_image and image_source is None:
        image_source = camera_image

if image_source is not None:

    # Save image source (upload or camera) to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(image_source.read())
        tmp_path = tmp.name

    # ────────────────────────────────────────────────
    # STEP 0A — Blur Detection
    # ────────────────────────────────────────────────
    img_gray = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is not None:
        blur_variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        if blur_variance < 40.0:  # Threshold for blurriness
            st.markdown(f"""
            <div style='background:var(--badge-B-bg);border:1px solid var(--badge-B-border);
                        border-radius:10px;padding:18px 24px;margin:12px 0'>
                <h4 style='color:var(--badge-B-fg);margin:0 0 6px 0'>⚠️ Blurry Image Detected</h4>
                <p style='color:var(--text-primary);margin:0 0 12px 0'>
                    The uploaded image appears to be out of focus (variance: {blur_variance:.1f}). 
                    For accurate defect detection, please ensure the PCB is well-lit and in focus.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Retake / Re-upload Image", use_container_width=True):
                st.session_state.uploader_key += 1
                st.session_state.camera_key += 1
                st.rerun()
            os.unlink(tmp_path)
            st.stop()

    # ────────────────────────────────────────────────
    # STEP 0B — PCB Gate Check (Groq)
    # ────────────────────────────────────────────────
    with st.spinner("🔎 Verifying image is a PCB (Groq)..."):
        is_pcb, pcb_conf, pcb_reason, pcb_model = groq_vision.is_pcb_image(tmp_path)

    if not is_pcb:
        # High-confidence rejection (>=80% — override already happened inside groq_vision.py)
        st.markdown(f"""
        <div style='background:var(--badge-R-bg);border:1px solid var(--badge-R-border);
                    border-radius:10px;padding:18px 24px;margin:12px 0'>
            <h4 style='color:var(--badge-R-fg);margin:0 0 6px 0'>&#10060; Not a PCB Image</h4>
            <p style='color:var(--text-primary);margin:0'>
                Groq validation rejected this image with
                <b>{pcb_conf*100:.0f}% confidence</b>.<br>
                <span style='color:var(--text-secondary)'>{pcb_reason}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        os.unlink(tmp_path)
        st.stop()
    elif pcb_conf == 0.0:
        # API failed — show warning but continue
        st.warning(f"⚠️ Groq PCB check skipped ({pcb_reason}). Proceeding with YOLO only.")
    elif "[Low-confidence rejection overridden" in pcb_reason:
        # Was flagged as non-PCB but with low confidence — show soft warning
        st.info(f"ℹ️ Groq was uncertain about this image ({pcb_conf*100:.0f}% confidence). Proceeding with inspection anyway.")

    # ────────────────────────────────────────────────
    # STEP 1 — Preprocessing (optional)
    # ────────────────────────────────────────────────
    if use_preprocessing and any([use_clahe, use_sharpen, use_denoise]):
        prep_bgr, prep_rgb = preprocess_pcb(
            tmp_path,
            use_clahe=use_clahe,       clahe_clip=clahe_clip, clahe_grid=clahe_grid,
            use_sharpen=use_sharpen,   sharpen_strength=sharpen_strength,
            use_denoise=use_denoise,   denoise_h=denoise_h
        )
        proc_path = tmp_path.replace('.jpg', '_proc.jpg')
        cv2.imwrite(proc_path, prep_bgr)
        model_input_path = proc_path
    else:
        prep_rgb         = None
        proc_path        = None
        model_input_path = tmp_path

    # ────────────────────────────────────────────────
    # STEP 2 — YOLO structural defect detection
    # ────────────────────────────────────────────────
    with st.spinner("🔬 Step 1/2 — YOLO structural analysis..."):
        result = grade_pcb(model_input_path, model, conf=conf_threshold, iou=iou_threshold)
        annotated_img = get_annotated_image_clean(
            model_input_path, model,
            conf=conf_threshold, iou=iou_threshold
        )

    # ────────────────────────────────────────────────
    # STEP 3 — Groq solder defect detection
    # ────────────────────────────────────────────────
    img_h_px, img_w_px = annotated_img.shape[:2]
    with st.spinner("🤖 Step 2/2 — Groq solder defect analysis..."):
        solder_defects, groq_err = groq_vision.detect_solder_defects(
            model_input_path, img_w_px, img_h_px
        )

    # Merge Groq solder defects into the result
    groq_defect_count = len(solder_defects)
    if solder_defects:
        # Annotate image with gold Groq boxes
        annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        annotated_bgr = groq_vision.draw_solder_defects(annotated_bgr, solder_defects)
        annotated_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Build unified defect entries compatible with grade_pcb output
        for sd in solder_defects:
            result['defects'].append({
                'class':      sd['class'],
                'confidence': sd['confidence'],
                'bbox':       sd['bbox'],
                'area':       sd['area'],
                'zone':       sd.get('zone', '—'),
                'source':     'groq',
            })
        result['defect_count'] += groq_defect_count

        # Re-score with merged defects
        from grading_engine import SEVERITY_WEIGHTS, GRADE_THRESHOLDS
        total_D = sum(
            SEVERITY_WEIGHTS.get(d['class'], 0.5) * d['confidence']
            for d in result['defects']
        )
        img_area = img_w_px * img_h_px
        total_area = sum(d.get('area', 0) for d in result['defects'])
        new_rho = round(total_area / img_area * 1e4, 4)
        new_score = max(0, round(100 - (0.40*total_D + 0.30*new_rho + 0.30*result['Z']) * 100, 1))
        result['D']     = round(total_D, 4)
        result['rho']   = new_rho
        result['score'] = new_score
        # Re-grade
        for grade_letter, threshold in sorted(GRADE_THRESHOLDS.items(),
                                             key=lambda x: x[1], reverse=True):
            if new_score >= threshold:
                result['grade'] = grade_letter
                break
        else:
            result['grade'] = 'Reject'

    # Save merged annotated image
    annotated_tmp = tmp_path.replace('.jpg', '_annotated.jpg')
    cv2.imwrite(annotated_tmp, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

    # ── Auto-log this inspection (only once per image) ───────
    if 'logged_images' not in st.session_state:
        st.session_state.logged_images = set()
        
    file_uid = getattr(image_source, "file_id", getattr(image_source, "name", "unknown"))
    
    if file_uid not in st.session_state.logged_images:
        _src = 'camera' if (camera_image is not None and image_source is camera_image) else 'upload'
        log_inspection(result, annotated_tmp, source=_src)
        st.session_state.logged_images.add(file_uid)

    # ── Groq status badge ────────────────────────────────
    if groq_err:
        st.warning(f"\u26a0\ufe0f Groq solder scan: {groq_err}")
    elif groq_defect_count > 0:
        st.markdown(f"""
        <div style='background:var(--bg-surface);border:1px solid var(--warning);
                    border-radius:8px;padding:10px 16px;margin:8px 0;
                    display:flex;align-items:center;gap:10px'>
            <span style='font-size:1.2rem'>&#x1F916;</span>
            <span style='color:var(--warning);font-weight:600'>Groq detected
            {groq_defect_count} additional solder defect{"s" if groq_defect_count>1 else ""}
            </span>
            <span style='color:var(--text-secondary);font-size:0.8rem'>(shown in gold boxes)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:var(--bg-surface);border:1px solid var(--success);
                    border-radius:8px;padding:10px 16px;margin:8px 0;
                    display:flex;align-items:center;gap:10px'>
            <span style='font-size:1.2rem'>&#x1F916;</span>
            <span style='color:var(--success);font-weight:600'>Groq: No solder defects detected</span>
        </div>
        """, unsafe_allow_html=True)


    # ── Images panel ────────────────────────────────────
    if prep_rgb is not None:
        # 3-panel: original | preprocessed | detected
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 📷 Original")
            st.image(Image.open(tmp_path), use_container_width=True)
        with col2:
            st.markdown("##### ✨ Preprocessed")
            st.image(prep_rgb, use_container_width=True)
        with col3:
            st.markdown("##### 🔍 Detected Defects")
            st.image(annotated_img, use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📷 Original Image")
            st.image(Image.open(tmp_path), use_container_width=True)
        with col2:
            st.markdown("##### 🔍 Detected Defects")
            st.image(annotated_img, use_container_width=True)

    st.divider()

    # ── Grade display ───────────────────────────────────
    grade = result['grade']
    score = result['score']

    grade_class = f"grade-{grade.replace(' ','')}"
    verdict_map = {
        'A':      '✅ Approve for shipping — meets quality standards.',
        'B':      '🔧 Minor rework recommended — re-inspect after fixing.',
        'C':      '⚠️  Mandatory rework before use.',
        'Reject': '❌ Critical defects — scrap or complete rework.'
    }

    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:24px; margin-bottom:8px'>
        <div class='grade-badge {grade_class}'>{grade}</div>
        <div>
            <div style='font-size:2rem; font-weight:800; color:var(--text-primary)'>{score}<span style='font-size:1rem; color:var(--text-secondary)'>/100</span></div>
            <div style='color:var(--text-secondary); font-size:0.85rem'>Quality Score</div>
        </div>
    </div>
    <div class='verdict-box'>{verdict_map.get(grade,'')}</div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Metrics row ─────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Defects Found",  result['defect_count'])
    m2.metric("Cleanliness",    f"{result['cleanliness']}%")
    m3.metric("Defect Score",   result['D'])
    m4.metric("Zone Penalty",   result['Z'])

    st.divider()

    # ── Defect table ────────────────────────────────────
    if result['defects']:
        st.markdown("##### \U0001f4cb Detected Defect Breakdown")
        _solder_sev = {
            'solder_bridge': 0.88, 'lifted_pad': 0.82, 'tombstone': 0.80,
            'solder_burn': 0.75,   'cold_joint': 0.70,  'solder_ball': 0.65,
            'solder_defect': 0.65, 'insufficient_solder': 0.60, 'excess_solder': 0.50,
        }
        df = pd.DataFrame([{
            "Defect Class":    d['class'],
            "Source":          d.get('source', 'yolo').upper(),
            "Confidence":      round(d['confidence'], 3),
            "Severity Weight": SEVERITY_WEIGHTS.get(d['class'],
                                   _solder_sev.get(d['class'], 0.65)),
            "Zone":            d.get('zone', '\u2014'),
            "Area %":          round(d.get('area_pct', 0), 3),
        } for d in result['defects']])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No defects detected — board is clean.")

    st.divider()

    # ── PDF Export ──────────────────────────────────────
    st.markdown("##### 📄 Export Report")
    pdf_bytes = generate_pdf(result, tmp_path, annotated_tmp)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name=f"AutoGrade_{grade}_{score}_{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # Cleanup
    try:
        os.unlink(tmp_path)
        os.unlink(annotated_tmp)
        if proc_path:
            os.unlink(proc_path)
    except:
        pass

else:
    # Empty state — themed
    st.markdown("""
    <div style='text-align:center; padding:60px 0'>
        <div style='font-size:4rem'>&#x1F52C;</div>
        <div style='font-size:1.2rem; margin-top:12px;
                    color:var(--text-primary); font-weight:600'>
            Upload a PCB image to begin inspection
        </div>
        <div style='font-size:0.9rem; margin-top:8px; color:var(--text-secondary)'>
            Supports JPG and PNG &nbsp;&middot;&nbsp; Works with real PCBs and dataset images
        </div>
    </div>
    """, unsafe_allow_html=True)