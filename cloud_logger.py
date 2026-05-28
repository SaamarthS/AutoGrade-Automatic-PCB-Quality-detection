"""
cloud_logger.py — Dual-backend inspection logger for AutoGrade.

  • On Streamlit Cloud (or any env with gcp_service_account in st.secrets):
      → Appends each inspection row to a Google Sheet.
  • Locally (or when Google credentials are absent):
      → Falls back to the original local CSV + image copy behaviour.

Usage in app.py:
    from cloud_logger import log_inspection
"""

import os
import datetime
import shutil
import pandas as pd

# ── Local log paths (used when running locally) ──────────────────────────────
_LOG_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
_LOG_IMAGES = os.path.join(_LOG_DIR, 'images')
_LOG_CSV    = os.path.join(_LOG_DIR, 'inspection_log.csv')


def _is_cloud() -> bool:
    """Return True if Google Sheets credentials are available via st.secrets."""
    try:
        import streamlit as st
        return "gcp_service_account" in st.secrets and "GSHEET_URL" in st.secrets
    except Exception:
        return False


def _get_sheet():
    """Return the first worksheet of the configured Google Sheet."""
    import streamlit as st
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=scopes,
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(st.secrets["GSHEET_URL"])
    return sh.sheet1


# Expected header row for the Google Sheet
_HEADERS = [
    "timestamp", "source", "grade", "score",
    "defect_count", "defect_classes",
    "cleanliness_%", "severity_D", "density_rho", "zone_penalty_Z",
]


def _ensure_headers(ws) -> None:
    """If the sheet is empty, write the header row."""
    try:
        first = ws.row_values(1)
        if not first or first[0] != "timestamp":
            ws.insert_row(_HEADERS, index=1)
    except Exception:
        pass


def log_inspection(result: dict, annotated_path: str, source: str = 'upload') -> None:
    """
    Log one inspection result.

    Parameters
    ----------
    result        : dict returned by grade_pcb()
    annotated_path: absolute path to the annotated JPEG on disk
    source        : 'upload' | 'camera'
    """
    now      = datetime.datetime.now()
    ts_file  = now.strftime('%Y%m%d_%H%M%S')
    ts_human = now.strftime('%Y-%m-%d %H:%M:%S')

    defect_cls = '|'.join(sorted(set(d['class'] for d in result['defects']))) or 'none'

    row_dict = {
        'timestamp':      ts_human,
        'source':         source,
        'grade':          result['grade'],
        'score':          result['score'],
        'defect_count':   result['defect_count'],
        'defect_classes': defect_cls,
        'cleanliness_%':  result['cleanliness'],
        'severity_D':     result['D'],
        'density_rho':    result['rho'],
        'zone_penalty_Z': result['Z'],
    }

    # ── Cloud path: write to Google Sheets ───────────────────────────────────
    if _is_cloud():
        try:
            ws = _get_sheet()
            _ensure_headers(ws)
            row_values = [str(row_dict.get(h, '')) for h in _HEADERS]
            ws.append_row(row_values, value_input_option='USER_ENTERED')
        except Exception as e:
            # Log to Streamlit but don't crash the app
            try:
                import streamlit as st
                st.warning(f"⚠️ Cloud log failed: {e}")
            except Exception:
                pass
        return

    # ── Local path: write to CSV + copy annotated image ──────────────────────
    os.makedirs(_LOG_IMAGES, exist_ok=True)

    img_name = f"{ts_file}_{result['grade']}_score{result['score']}.jpg"
    img_dest = os.path.join(_LOG_IMAGES, img_name)
    try:
        shutil.copy2(annotated_path, img_dest)
    except Exception:
        img_name = 'error_copying_image'

    row_dict['annotated_image'] = img_name

    file_exists = os.path.isfile(_LOG_CSV)
    pd.DataFrame([row_dict]).to_csv(
        _LOG_CSV, mode='a', header=not file_exists, index=False
    )


def load_log() -> pd.DataFrame:
    """
    Return the full inspection history as a DataFrame.

    On cloud → reads from Google Sheets.
    Locally  → reads from the local CSV.
    Returns an empty DataFrame (with correct columns) if no data yet.
    """
    if _is_cloud():
        try:
            ws   = _get_sheet()
            data = ws.get_all_records()
            return pd.DataFrame(data) if data else pd.DataFrame(columns=_HEADERS)
        except Exception:
            return pd.DataFrame(columns=_HEADERS)

    if os.path.isfile(_LOG_CSV):
        return pd.read_csv(_LOG_CSV)
    return pd.DataFrame(columns=_HEADERS + ['annotated_image'])
