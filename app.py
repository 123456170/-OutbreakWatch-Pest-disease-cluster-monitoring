# app.py
# OutbreakWatch - Streamlit app for farm pest/disease report tracking,
# automatic cluster detection, Gemini-powered analysis/advisories, heatmaps,
# spread timeline, alerts, and weekly PDF bulletin.
#
# Runs fully without a Gemini API key in "offline demo AI" mode.
# If GEMINI_API_KEY is present in st.secrets or environment, real Gemini is used.

import os
import io
import re
import json
import uuid
import random
import sqlite3
import pathlib
from datetime import date, datetime, timedelta
from contextlib import closing

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image

try:
    import google.generativeai as genai
except Exception:
    genai = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =============================================================================
# Global constants
# =============================================================================

BASE_DIR = pathlib.Path(__file__).parent.resolve() if "__file__" in locals() else pathlib.Path.cwd()
DB_PATH = BASE_DIR / "outbreakwatch.db"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

GEMINI_MODEL = "gemini-2.0-flash"

ISSUE_TYPES = [
    "insect pest",
    "fungal",
    "bacterial",
    "viral",
    "nutrient deficiency",
    "abiotic",
    "unknown",
]

COMMON_CROPS = [
    "Tomato",
    "Potato",
    "Chili",
    "Onion",
    "Cotton",
    "Wheat",
    "Maize",
    "Rice",
    "Banana",
    "Apple",
    "Grape",
    "Other",
]

LANG_OPTIONS = ["English", "اردو", "हिंदी"]
LANG_CODE = {
    "English": "en",
    "اردو": "ur",
    "हिंदी": "hi",
}
CODE_LANG = {v: k for k, v in LANG_CODE.items()}

ADV_LANG_NAMES = {
    "en": "English",
    "ur": "Urdu (Urdu script)",
    "hi": "Hindi (Devanagari script)",
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem;}
    div[data-testid="stMetric"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Database helpers
# =============================================================================

def get_conn():
    """Return SQLite connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create database tables if they do not exist."""
    with closing(get_conn()) as conn, conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                row_label TEXT,
                col_label TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_path TEXT,
                crop TEXT,
                zone TEXT,
                date TEXT,
                reporter TEXT,
                issue TEXT,
                issue_type TEXT,
                severity INTEGER,
                confidence REAL,
                visible_symptoms TEXT,
                likely_spread_mode TEXT,
                first_actions_non_chemical TEXT,
                first_actions_chemical TEXT,
                raw_json TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule TEXT,
                issue TEXT,
                zones TEXT,
                created_at TEXT,
                acknowledged INTEGER DEFAULT 0
            );
            """
        )


def save_zones(rows, cols, custom_names=None):
    """
    Save a zone grid.
    Default naming is A1, A2, ... D6.
    Optional custom names can be supplied comma-separated in row-major order.
    """
    rows = max(1, min(26, int(rows)))
    cols = max(1, min(50, int(cols)))

    names = []
    if custom_names:
        parts = [x.strip() for x in str(custom_names).split(",") if x.strip()]
        if len(parts) == rows * cols:
            names = parts

    with closing(get_conn()) as conn, conn:
        conn.execute("DELETE FROM zones")
        idx = 0
        now = datetime.now().isoformat(sep=" ")
        for r in range(rows):
            row_label = chr(ord("A") + r)
            for c in range(1, cols + 1):
                name = names[idx] if idx < len(names) else f"{row_label}{c}"
                conn.execute(
                    "INSERT OR REPLACE INTO zones(name, row_label, col_label, created_at) VALUES(?,?,?,?)",
                    (name, row_label, str(c), now),
                )
                idx += 1


def load_zones():
    """Load zones as DataFrame."""
    with closing(get_conn()) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM zones ORDER BY row_label, CAST(col_label AS INTEGER)",
            conn,
        )
    return df


def zone_names():
    """Return list of zone names for dropdowns."""
    zones = load_zones()
    if zones.empty:
        return []
    return zones["name"].tolist()


def load_reports():
    """Load all reports as DataFrame."""
    with closing(get_conn()) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM reports ORDER BY date DESC, id DESC",
            conn,
        )
    if not df.empty:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def insert_report(record: dict) -> int:
    """Insert one report row."""
    sql = """
        INSERT INTO reports(
            photo_path, crop, zone, date, reporter, issue, issue_type,
            severity, confidence, visible_symptoms, likely_spread_mode,
            first_actions_non_chemical, first_actions_chemical, raw_json, created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    vals = (
        record.get("photo_path"),
        record.get("crop"),
        record.get("zone"),
        record.get("date"),
        record.get("reporter"),
        record.get("issue"),
        record.get("issue_type"),
        record.get("severity"),
        record.get("confidence"),
        record.get("visible_symptoms"),
        record.get("likely_spread_mode"),
        record.get("first_actions_non_chemical"),
        record.get("first_actions_chemical"),
        record.get("raw_json"),
        datetime.now().isoformat(sep=" "),
    )
    with closing(get_conn()) as conn, conn:
        cur = conn.execute(sql, vals)
        return int(cur.lastrowid)


def active_alerts_df():
    """Return active alerts."""
    with closing(get_conn()) as conn:
        return pd.read_sql_query(
            "SELECT * FROM alerts WHERE acknowledged=0 ORDER BY created_at DESC",
            conn,
        )


def all_alerts_df():
    """Return all alerts, active first."""
    with closing(get_conn()) as conn:
        return pd.read_sql_query(
            "SELECT * FROM alerts ORDER BY acknowledged ASC, created_at DESC",
            conn,
        )


def acknowledge_alert(alert_id: int):
    """Mark alert acknowledged."""
    with closing(get_conn()) as conn, conn:
        conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (int(alert_id),))


def insert_alert(rule: str, issue: str, zones_list) -> bool:
    """
    Insert alert if the same alert was not already inserted today.
    This avoids duplicate alerts on every rerun.
    """
    zones_str = ", ".join(sorted(set([str(z) for z in zones_list if z])))
    now = datetime.now().isoformat(sep=" ")
    today = date.today().isoformat()

    with closing(get_conn()) as conn, conn:
        exists = conn.execute(
            """
            SELECT id FROM alerts
            WHERE rule=? AND issue=? AND zones=? AND date(created_at)=?
            """,
            (rule, issue, zones_str, today),
        ).fetchone()
        if exists:
            return False

        conn.execute(
            "INSERT INTO alerts(rule, issue, zones, created_at, acknowledged) VALUES(?,?,?,?,0)",
            (rule, issue, zones_str, now),
        )
        return True


def clear_all_data():
    """Clear reports and alerts, keep zones."""
    with closing(get_conn()) as conn, conn:
        conn.execute("DELETE FROM reports")
        conn.execute("DELETE FROM alerts")


# =============================================================================
# Outbreak rule engine
# =============================================================================

def run_outbreak_rules() -> int:
    """
    Outbreak rules:
    1) Same issue_name in >=3 reports across >=2 zones within 7 days.
    2) Average severity >=4 in any zone for any issue within 7 days.
    Returns number of newly inserted alerts.
    """
    df = load_reports()
    if df.empty:
        return 0

    df = df.dropna(subset=["date_dt"])
    if df.empty:
        return 0

    end_date = max(date.today(), df["date_dt"].dt.date.max())
    start_date = end_date - timedelta(days=6)

    recent = df[
        (df["date_dt"].dt.date >= start_date) &
        (df["date_dt"].dt.date <= end_date)
    ]

    created = 0

    if not recent.empty:
        # Rule 1: cluster across zones
        for issue, group in recent.groupby("issue"):
            if not issue or str(issue).strip().lower() in ["", "unknown", "unknown disorder"]:
                continue
            zones = [z for z in group["zone"].dropna().unique().tolist() if z]
            if len(group) >= 3 and len(zones) >= 2:
                if insert_alert("cluster_3_reports_2_zones_7d", str(issue), zones):
                    created += 1

        # Rule 2: average severity >=4 in a zone
        sev = recent.groupby(["zone", "issue"])["severity"].mean()
        for (zone, issue), avg in sev.items():
            if not zone or not issue or str(issue).strip().lower() in ["", "unknown", "unknown disorder"]:
                continue
            if float(avg) >= 4.0:
                if insert_alert("average_severity_ge_4_zone", str(issue), [zone]):
                    created += 1

    return created


# =============================================================================
# Demo seed data
# =============================================================================

def seed_demo_data(force: bool = False) -> bool:
    """
    Seed realistic demo reports and trigger outbreak rules.
    If data already exists and force=False, do nothing.
    """
    with closing(get_conn()) as conn:
        report_count = conn.execute("SELECT COUNT(*) AS c FROM reports").fetchone()["c"]

    if report_count > 0 and not force:
        return False

    if force:
        with closing(get_conn()) as conn, conn:
            conn.execute("DELETE FROM reports")
            conn.execute("DELETE FROM alerts")

    if load_zones().empty or force:
        save_zones(4, 6)

    zone_list = zone_names()
    if not zone_list:
        zone_list = [f"{chr(65+r)}{c}" for r in range(4) for c in range(1, 7)]

    rows = []

    def add_report(
        crop,
        zone,
        days_ago,
        issue,
        issue_type,
        severity,
        confidence,
        symptoms,
        spread,
        nonchem,
        chem,
        reporter="Demo Scout",
    ):
        d = (date.today() - timedelta(days=int(days_ago))).isoformat()
        rows.append(
            (
                None,
                crop,
                zone,
                d,
                reporter,
                issue,
                issue_type,
                int(severity),
                float(confidence),
                json.dumps(symptoms, ensure_ascii=False),
                spread,
                json.dumps(nonchem, ensure_ascii=False),
                json.dumps(chem, ensure_ascii=False),
                json.dumps({"source": "demo"}, ensure_ascii=False),
                datetime.now().isoformat(sep=" "),
            )
        )

    # Cluster 1: Tomato leaf curl virus, multiple zones, high severity
    leaf_symptoms = ["Leaf curling", "Yellowing", "Stunted growth"]
    leaf_nonchem = ["Remove infected plants", "Use virus-free seedlings", "Install yellow sticky traps for whitefly"]
    leaf_chem = ["Insecticides for whitefly control"]
    add_report("Tomato", "A2", 0, "Tomato leaf curl virus", "viral", 4, 0.86,
               leaf_symptoms, "Whitefly vectors and infected seedlings", leaf_nonchem, leaf_chem, "Field scout A")
    add_report("Tomato", "A3", 1, "Tomato leaf curl virus", "viral", 4, 0.83,
               leaf_symptoms, "Whitefly vectors and infected seedlings", leaf_nonchem, leaf_chem, "Field scout A")
    add_report("Tomato", "B2", 1, "Tomato leaf curl virus", "viral", 5, 0.88,
               leaf_symptoms, "Whitefly vectors and infected seedlings", leaf_nonchem, leaf_chem, "Field scout B")
    add_report("Tomato", "A2", 2, "Tomato leaf curl virus", "viral", 4, 0.82,
               leaf_symptoms, "Whitefly vectors and infected seedlings", leaf_nonchem, leaf_chem, "Farmer input")
    add_report("Tomato", "B2", 3, "Tomato leaf curl virus", "viral", 5, 0.87,
               leaf_symptoms, "Whitefly vectors and infected seedlings", leaf_nonchem, leaf_chem, "Field scout B")
    add_report("Tomato", "A3", 4, "Tomato leaf curl virus", "viral", 4, 0.81,
               leaf_symptoms, "Whitefly vectors and infected seedlings", leaf_nonchem, leaf_chem, "Farmer input")

    # Cluster 2: Early blight in C/D zones
    early_symptoms = ["Dark leaf spots", "Yellow halo", "Lower leaf infection"]
    early_nonchem = ["Remove infected leaves", "Improve air flow", "Avoid overhead irrigation"]
    early_chem = ["Protective fungicides", "Curative fungicides"]
    add_report("Tomato", "C3", 0, "Early blight", "fungal", 3, 0.79,
               early_symptoms, "Airborne spores and infected debris", early_nonchem, early_chem, "Field scout C")
    add_report("Tomato", "C4", 1, "Early blight", "fungal", 4, 0.81,
               early_symptoms, "Airborne spores and infected debris", early_nonchem, early_chem, "Field scout C")
    add_report("Potato", "D4", 2, "Early blight", "fungal", 3, 0.77,
               early_symptoms, "Airborne spores and infected debris", early_nonchem, early_chem, "Farmer input")
    add_report("Tomato", "C3", 4, "Early blight", "fungal", 4, 0.80,
               early_symptoms, "Airborne spores and infected debris", early_nonchem, early_chem, "Field scout C")

    # Cluster 3: Aphids in A/B zones
    aphid_symptoms = ["Curled leaves", "Sticky honeydew", "Colonies on new growth"]
    aphid_nonchem = ["Spray water jet", "Encourage ladybirds", "Remove weeds"]
    aphid_chem = ["Contact insecticides", "Systemic insecticides"]
    add_report("Chili", "A1", 0, "Aphids", "insect pest", 2, 0.74,
               aphid_symptoms, "Winged aphids and ants", aphid_nonchem, aphid_chem, "Field scout A")
    add_report("Chili", "B1", 2, "Aphids", "insect pest", 3, 0.76,
               aphid_symptoms, "Winged aphids and ants", aphid_nonchem, aphid_chem, "Farmer input")
    add_report("Tomato", "B2", 3, "Aphids", "insect pest", 4, 0.78,
               aphid_symptoms, "Winged aphids and ants", aphid_nonchem, aphid_chem, "Field scout B")
    add_report("Chili", "A1", 5, "Aphids", "insect pest", 3, 0.73,
               aphid_symptoms, "Winged aphids and ants", aphid_nonchem, aphid_chem, "Field scout A")

    # Cluster 4: Bacterial wilt, severe in D zones
    bac_symptoms = ["Sudden wilting", "Brown vascular streaks", "Plant collapse"]
    bac_nonchem = ["Remove infected plants", "Avoid waterlogging", "Sanitize tools"]
    bac_chem = ["Copper-based bactericides"]
    add_report("Tomato", "D1", 0, "Bacterial wilt", "bacterial", 5, 0.84,
               bac_symptoms, "Soil, water, and contaminated tools", bac_nonchem, bac_chem, "Field scout D")
    add_report("Tomato", "D2", 1, "Bacterial wilt", "bacterial", 4, 0.82,
               bac_symptoms, "Soil, water, and contaminated tools", bac_nonchem, bac_chem, "Field scout D")
    add_report("Chili", "D1", 2, "Bacterial wilt", "bacterial", 5, 0.85,
               bac_symptoms, "Soil, water, and contaminated tools", bac_nonchem, bac_chem, "Farmer input")

    # Nutrient deficiency
    nit_symptoms = ["Pale older leaves", "Slow growth"]
    nit_nonchem = ["Apply compost or manure", "Do soil test"]
    nit_chem = ["Nitrogen fertilizer classes"]
    add_report("Maize", "B4", 3, "Nitrogen deficiency", "nutrient deficiency", 2, 0.70,
               nit_symptoms, "Not infectious; field nutrition problem", nit_nonchem, nit_chem, "Farmer input")
    add_report("Maize", "C4", 6, "Nitrogen deficiency", "nutrient deficiency", 2, 0.69,
               nit_symptoms, "Not infectious; field nutrition problem", nit_nonchem, nit_chem, "Field scout C")

    # Powdery mildew
    pow_symptoms = ["White powdery patches", "Leaf distortion"]
    pow_nonchem = ["Prune shaded leaves", "Improve airflow"]
    pow_chem = ["Protective fungicides", "Curative fungicides"]
    add_report("Grape", "A4", 1, "Powdery mildew", "fungal", 3, 0.75,
               pow_symptoms, "Airborne spores", pow_nonchem, pow_chem, "Field scout A")
    add_report("Grape", "B4", 5, "Powdery mildew", "fungal", 3, 0.74,
               pow_symptoms, "Airborne spores", pow_nonchem, pow_chem, "Farmer input")

    # Background older reports
    random.seed(42)
    background = [
        ("Tomato", "Late blight", "fungal", 3, 0.72, ["Dark water-soaked lesions"], "Airborne spores", ["Remove infected leaves"], ["Protective fungicides"]),
        ("Potato", "Late blight", "fungal", 3, 0.73, ["Dark lesions on leaves"], "Airborne spores", ["Remove infected leaves"], ["Protective fungicides"]),
        ("Wheat", "Leaf rust", "fungal", 2, 0.70, ["Orange pustules"], "Windborne spores", ["Plant resistant varieties"], ["Protective fungicides"]),
        ("Cotton", "Aphids", "insect pest", 2, 0.68, ["Curled leaves"], "Winged aphids", ["Remove weeds"], ["Contact insecticides"]),
        ("Maize", "Fall armyworm", "insect pest", 3, 0.74, ["Whorl feeding"], "Moth flight", ["Handpick larvae"], ["Biopesticides", "Contact insecticides"]),
        ("Banana", "Banana leaf spot", "fungal", 3, 0.71, ["Brown leaf spots"], "Rain-splashed spores", ["Remove affected leaves"], ["Protective fungicides"]),
    ]
    for i in range(18):
        crop, issue, issue_type, severity, confidence, symptoms, spread, nonchem, chem = random.choice(background)
        zone = random.choice(zone_list)
        days_ago = random.randint(8, 20)
        add_report(crop, zone, days_ago, issue, issue_type, severity, confidence,
                   symptoms, spread, nonchem, chem, reporter="Historical scout")

    with closing(get_conn()) as conn, conn:
        conn.executemany(
            """
            INSERT INTO reports(
                photo_path, crop, zone, date, reporter, issue, issue_type,
                severity, confidence, visible_symptoms, likely_spread_mode,
                first_actions_non_chemical, first_actions_chemical, raw_json, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )

    run_outbreak_rules()
    return True


# =============================================================================
# Gemini / AI helpers
# =============================================================================

def get_api_key():
    """Read Gemini API key from Streamlit secrets or environment. Never hardcoded."""
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None

    if not key:
        key = os.getenv("GEMINI_API_KEY", "").strip()

    return key


def gemini_available() -> bool:
    """True if library and API key are available."""
    return genai is not None and bool(get_api_key())


def gemini_model():
    """Configure and return Gemini model."""
    genai.configure(api_key=get_api_key())
    return genai.GenerativeModel(GEMINI_MODEL)


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences and extract first JSON object if present."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    if "```" in text:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    return text


def as_list(value):
    """Coerce Gemini list fields into list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,;،]", value)
        return [p.strip() for p in parts if p.strip()]
    return [str(value).strip()]


def validate_analysis_json(raw_text: str) -> dict:
    """
    Validate Gemini vision JSON response.
    Required keys, types, allowed issue_type, severity 1-5, confidence 0-1.
    """
    cleaned = strip_code_fences(raw_text)
    data = json.loads(cleaned)

    if not isinstance(data, dict):
        raise ValueError("JSON response must be an object.")

    required = [
        "issue_name",
        "issue_type",
        "severity",
        "confidence",
        "visible_symptoms",
        "likely_spread_mode",
        "first_actions_non_chemical",
        "first_actions_chemical",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    data["issue_name"] = str(data.get("issue_name") or "unknown").strip() or "unknown"

    issue_type_raw = str(data.get("issue_type") or "unknown").strip().lower()
    if issue_type_raw in ISSUE_TYPES:
        issue_type = issue_type_raw
    else:
        issue_type = "unknown"
        aliases = {
            "insect": "insect pest",
            "pest": "insect pest",
            "fungus": "fungal",
            "fungal disease": "fungal",
            "bacteria": "bacterial",
            "virus": "viral",
            "nutrient": "nutrient deficiency",
            "deficiency": "nutrient deficiency",
            "abiotic stress": "abiotic",
        }
        for key, val in aliases.items():
            if key in issue_type_raw:
                issue_type = val
                break

    data["issue_type"] = issue_type

    try:
        severity = int(float(str(data.get("severity", 3))))
    except Exception:
        severity = 3
    data["severity"] = max(1, min(5, severity))

    try:
        confidence = float(str(data.get("confidence", 0.5)))
    except Exception:
        confidence = 0.5
    data["confidence"] = max(0.0, min(1.0, confidence))

    data["visible_symptoms"] = as_list(data.get("visible_symptoms"))
    data["likely_spread_mode"] = str(data.get("likely_spread_mode") or "unknown").strip()
    data["first_actions_non_chemical"] = as_list(data.get("first_actions_non_chemical"))
    data["first_actions_chemical"] = as_list(data.get("first_actions_chemical"))

    return data


def mock_analysis(crop: str, notes: str) -> dict:
    """
    Offline demo classifier used when no Gemini API key is present or API fails.
    It returns the same strict JSON-like structure.
    """
    notes = (notes or "").lower()
    crop_l = (crop or "").lower()

    if "curl" in notes or "leaf curl" in notes:
        return {
            "issue_name": "Tomato leaf curl virus",
            "issue_type": "viral",
            "severity": 4,
            "confidence": 0.78,
            "visible_symptoms": ["Leaf curling", "Yellowing", "Stunted growth"],
            "likely_spread_mode": "Whitefly vectors and infected seedlings",
            "first_actions_non_chemical": ["Remove infected plants", "Use virus-free seedlings", "Install yellow sticky traps"],
            "first_actions_chemical": ["Insecticides for whitefly control"],
        }

    if "blight" in notes:
        return {
            "issue_name": "Early blight",
            "issue_type": "fungal",
            "severity": 3,
            "confidence": 0.76,
            "visible_symptoms": ["Dark leaf spots", "Yellow halo"],
            "likely_spread_mode": "Airborne spores and infected debris",
            "first_actions_non_chemical": ["Remove infected leaves", "Avoid overhead irrigation"],
            "first_actions_chemical": ["Protective fungicides"],
        }

    if "aphid" in notes:
        return {
            "issue_name": "Aphids",
            "issue_type": "insect pest",
            "severity": 3,
            "confidence": 0.74,
            "visible_symptoms": ["Curled leaves", "Sticky honeydew"],
            "likely_spread_mode": "Winged aphids and ants",
            "first_actions_non_chemical": ["Spray water jet", "Encourage ladybirds"],
            "first_actions_chemical": ["Contact insecticides"],
        }

    if "mildew" in notes or "powder" in notes:
        return {
            "issue_name": "Powdery mildew",
            "issue_type": "fungal",
            "severity": 3,
            "confidence": 0.73,
            "visible_symptoms": ["White powdery patches", "Leaf distortion"],
            "likely_spread_mode": "Airborne spores",
            "first_actions_non_chemical": ["Prune shaded leaves", "Improve airflow"],
            "first_actions_chemical": ["Protective fungicides"],
        }

    if "wilt" in notes:
        return {
            "issue_name": "Bacterial wilt",
            "issue_type": "bacterial",
            "severity": 4,
            "confidence": 0.77,
            "visible_symptoms": ["Sudden wilting", "Brown vascular streaks"],
            "likely_spread_mode": "Soil, water, and contaminated tools",
            "first_actions_non_chemical": ["Remove infected plants", "Avoid waterlogging"],
            "first_actions_chemical": ["Copper-based bactericides"],
        }

    if "yellow" in notes or "nitrogen" in notes or "pale" in notes:
        return {
            "issue_name": "Nitrogen deficiency",
            "issue_type": "nutrient deficiency",
            "severity": 2,
            "confidence": 0.70,
            "visible_symptoms": ["Pale older leaves", "Slow growth"],
            "likely_spread_mode": "Not infectious; field nutrition problem",
            "first_actions_non_chemical": ["Apply compost or manure", "Do soil test"],
            "first_actions_chemical": ["Nitrogen fertilizer classes"],
        }

    if "tomato" in crop_l:
        return {
            "issue_name": "Early blight",
            "issue_type": "fungal",
            "severity": 3,
            "confidence": 0.72,
            "visible_symptoms": ["Dark leaf spots", "Yellowing lower leaves"],
            "likely_spread_mode": "Airborne spores and infected debris",
            "first_actions_non_chemical": ["Remove infected leaves", "Improve air flow"],
            "first_actions_chemical": ["Protective fungicides"],
        }

    if "potato" in crop_l:
        return {
            "issue_name": "Late blight",
            "issue_type": "fungal",
            "severity": 4,
            "confidence": 0.73,
            "visible_symptoms": ["Dark water-soaked lesions", "White mold on leaf undersides"],
            "likely_spread_mode": "Airborne spores",
            "first_actions_non_chemical": ["Remove infected foliage", "Avoid overhead irrigation"],
            "first_actions_chemical": ["Protective fungicides", "Curative fungicides"],
        }

    if "wheat" in crop_l:
        return {
            "issue_name": "Leaf rust",
            "issue_type": "fungal",
            "severity": 3,
            "confidence": 0.71,
            "visible_symptoms": ["Orange pustules on leaves"],
            "likely_spread_mode": "Windborne spores",
            "first_actions_non_chemical": ["Plant resistant varieties", "Monitor nearby fields"],
            "first_actions_chemical": ["Protective fungicides"],
        }

    if "cotton" in crop_l:
        return {
            "issue_name": "Aphids",
            "issue_type": "insect pest",
            "severity": 3,
            "confidence": 0.70,
            "visible_symptoms": ["Curled leaves", "Honeydew"],
            "likely_spread_mode": "Winged aphids",
            "first_actions_non_chemical": ["Remove weeds", "Encourage natural predators"],
            "first_actions_chemical": ["Contact insecticides"],
        }

    if "maize" in crop_l or "corn" in crop_l:
        return {
            "issue_name": "Fall armyworm",
            "issue_type": "insect pest",
            "severity": 3,
            "confidence": 0.72,
            "visible_symptoms": ["Whorl feeding", "Frass in whorl"],
            "likely_spread_mode": "Moth flight",
            "first_actions_non_chemical": ["Handpick larvae", "Install pheromone traps"],
            "first_actions_chemical": ["Biopesticides", "Contact insecticides"],
        }

    if "banana" in crop_l:
        return {
            "issue_name": "Banana leaf spot",
            "issue_type": "fungal",
            "severity": 3,
            "confidence": 0.71,
            "visible_symptoms": ["Brown leaf spots", "Leaf drying"],
            "likely_spread_mode": "Rain-splashed spores",
            "first_actions_non_chemical": ["Remove affected leaves", "Improve field sanitation"],
            "first_actions_chemical": ["Protective fungicides"],
        }

    return {
        "issue_name": "Unknown disorder",
        "issue_type": "unknown",
        "severity": 2,
        "confidence": 0.50,
        "visible_symptoms": ["Visible abnormality reported"],
        "likely_spread_mode": "Unknown",
        "first_actions_non_chemical": ["Mark the plant", "Monitor neighboring plants", "Consult local extension officer if spreading"],
        "first_actions_chemical": [],
    }


def analyze_with_ai(image_bytes, crop: str, zone: str, notes: str) -> dict:
    """
    Analyze report with Gemini vision if available.
    Falls back to offline demo classifier if no key or API fails.
    Retries once on invalid JSON.
    """
    if not gemini_available():
        return mock_analysis(crop, notes)

    prompt = f"""
You are a plant protection expert.
Analyze the crop issue and return ONLY a valid JSON object.
Do not use markdown, code fences, or comments.

Required JSON keys:
issue_name, issue_type, severity, confidence, visible_symptoms,
likely_spread_mode, first_actions_non_chemical, first_actions_chemical.

Rules:
- issue_type must be one of: insect pest, fungal, bacterial, viral, nutrient deficiency, abiotic, unknown.
- severity must be an integer from 1 to 5.
- confidence must be a number from 0 to 1.
- visible_symptoms must be a list of strings.
- first_actions_non_chemical must be a list of strings.
- first_actions_chemical must be a list of product classes only, NO doses.
- If chemical classes are listed, include a note in first_actions_chemical such as: "Follow label and confirm with local agriculture officer".
- If uncertain, use issue_name "Unknown disorder" and issue_type "unknown".

Context:
Crop: {crop}
Zone: {zone}
Reporter notes: {notes}
""".strip()

    try:
        model = gemini_model()
        content = [prompt]

        if image_bytes:
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img.load()
                content.append(img)
            except Exception:
                content.append(image_bytes)

        last_error = None
        for attempt in range(2):
            try:
                response = model.generate_content(content)
                raw = getattr(response, "text", "")
                return validate_analysis_json(raw)
            except Exception as e:
                last_error = e
                content[0] = prompt + "\nYour previous response was invalid or empty. Return ONLY valid JSON."

        raise last_error or ValueError("Gemini analysis failed.")

    except Exception as exc:
        st.warning(f"Gemini vision call failed. Using offline demo analysis. Detail: {exc}")
        return mock_analysis(crop, notes)


# =============================================================================
# Advisory generation
# =============================================================================

def json_list_from_db(value):
    """Parse JSON list stored in DB; fallback to single-item list."""
    if not value:
        return []
    try:
        data = json.loads(value)
        return data if isinstance(data, list) else [str(data)]
    except Exception:
        return [str(value)]


def get_issue_meta(issue: str):
    """Get latest known metadata for an issue from reports table."""
    with closing(get_conn()) as conn:
        row = conn.execute(
            """
            SELECT issue_type, likely_spread_mode, first_actions_non_chemical
            FROM reports
            WHERE issue=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (issue,),
        ).fetchone()

    if not row:
        return {}

    return {
        "issue_type": row["issue_type"],
        "likely_spread_mode": row["likely_spread_mode"],
        "first_actions_non_chemical": json_list_from_db(row["first_actions_non_chemical"]),
    }


def fallback_advisory(alert: dict, lang_code: str) -> str:
    """
    Built-in advisory if Gemini is unavailable or fails.
    Localized in English, Urdu, Hindi.
    """
    issue = str(alert.get("issue", "unknown issue"))
    zones = str(alert.get("zones", "multiple zones"))
    meta = get_issue_meta(issue)

    issue_type = meta.get("issue_type", "unknown")
    spread = meta.get("likely_spread_mode") or "wind, insects, tools, or infected plant material"
    actions = meta.get("first_actions_non_chemical") or []
    check = actions[0] if actions else "inspect plants and remove severely affected parts"

    if lang_code == "ur":
        return (
            f"کسان کے لیے انتباہ: {zones} میں {issue} ({issue_type}) کا خطرہ ہے۔ "
            f"یہ عام طور پر {spread} سے پھیلتا ہے۔ آج ہی یہ چیک کریں: {check}۔ "
            "اگر علامات تیزی سے بڑھیں، فصل شدید متاثر ہو، یا پیداوار کم ہونے لگے تو فوری طور پر مقامی زرعی ماہر سے رابطہ کریں۔ "
            "کیمیائی علاج صرف ضرورت پر کریں؛ لیبل والی مصنوعات استعمال کریں اور خوراک/طریقہ مقامی زرعی افسر سے تصدیق کریں۔"
        )

    if lang_code == "hi":
        return (
            f"किसानों के लिए सलाह: {zones} में {issue} ({issue_type}) का खतरा है। "
            f"यह आम तौर पर {spread} से फैलता है। आज जाँचें: {check}। "
            "अगर लक्षण तेजी से बढ़ें, फसल गंभीर रूप से प्रभावित हो, या उपज घटने लगे तो तुरंत स्थानीय कृषि विशेषज्ञ से संपर्क करें। "
            "रासायनिक उपचार केवल आवश्यकता पर करें; लेबल वाली उत्पाद श्रेणियाँ उपयोग करें और खुराक/तरीका स्थानीय कृषि अधिकारी से पुष्टि करें।"
        )

    return (
        f"Farmer advisory: {issue} ({issue_type}) has been flagged in {zones}. "
        f"It usually spreads by {spread}. Today, check nearby plants and {check.lower()}. "
        "Call an expert if symptoms are spreading quickly, severity is high, or yield is at risk. "
        "If chemicals are needed, use only labeled product classes and follow the label; confirm with the local agriculture officer. No doses are given here."
    )


def generate_advisory_text(alert: dict, lang_code: str) -> str:
    """
    Generate farmer advisory for an alert.
    Uses Gemini if available, otherwise fallback template.
    Cached per alert/language in Streamlit session_state.
    """
    alert_id = alert.get("id") or abs(hash((str(alert.get("issue")), str(alert.get("zones")))))
    cache_key = f"advisory_{alert_id}_{lang_code}"

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    issue = str(alert.get("issue", "unknown issue"))
    zones = str(alert.get("zones", "multiple zones"))
    rule = str(alert.get("rule", "outbreak rule"))
    meta = get_issue_meta(issue)
    spread = meta.get("likely_spread_mode") or "wind, insects, tools, or infected plant material"

    if gemini_available():
        try:
            model = gemini_model()
            language_name = ADV_LANG_NAMES.get(lang_code, "English")
            prompt = f"""
Write a short plain-language farmer advisory in {language_name}.
No markdown. No bullet points unless absolutely necessary.
Keep it under 120 words.

Alert rule: {rule}
Issue: {issue}
Zones: {zones}
Known spread mode: {spread}

Include:
1. What it is.
2. How it spreads.
3. What the farmer should check today.
4. When to call an expert.

If chemicals are mentioned, mention only product classes, never doses.
Say to follow the label and confirm with the local agriculture officer.
""".strip()

            for attempt in range(2):
                response = model.generate_content(prompt)
                text = getattr(response, "text", "").strip()
                if text:
                    text = text.strip("`").strip()
                    st.session_state[cache_key] = text
                    return text
                prompt += "\nYour previous answer was empty. Return plain text only."

        except Exception as exc:
            st.warning(f"Gemini advisory failed. Using built-in advisory. Detail: {exc}")

    text = fallback_advisory(alert, lang_code)
    st.session_state[cache_key] = text
    return text


# =============================================================================
# Heatmap / visualization helpers
# =============================================================================

def parse_zone_name(name):
    """Parse zone names like A1, B12, C-3 into (row_label, col_number)."""
    if not name:
        return None
    m = re.match(r"^([A-Za-z]+)\s*-?\s*(\d+)$", str(name).strip())
    if m:
        return m.group(1).upper(), int(m.group(2))
    return None


def zone_grid_data(df: pd.DataFrame):
    """
    Build a 2D matrix of report counts by zone.
    Uses zones table row/col labels if available; otherwise derives from report zone names.
    Returns matrix, row_labels, col_labels.
    """
    zones_df = load_zones()
    counts = {} if df.empty else df.groupby("zone").size().to_dict()

    if not zones_df.empty:
        row_labels = sorted([str(x) for x in zones_df["row_label"].dropna().unique().tolist()])

        col_labels_raw = zones_df["col_label"].dropna().unique().tolist()
        col_labels = sorted([
            int(float(str(x)))
            for x in col_labels_raw
            if str(x).replace(".", "", 1).isdigit()
        ])

        if not row_labels or not col_labels:
            parsed = [parse_zone_name(z) for z in counts.keys()]
            row_labels = sorted({p[0] for p in parsed if p})
            col_labels = sorted({p[1] for p in parsed if p})

        zone_pos = {}
        for _, zrow in zones_df.iterrows():
            try:
                r = row_labels.index(str(zrow["row_label"]))
                c = col_labels.index(int(float(str(zrow["col_label"]))))
                zone_pos[zrow["name"]] = (r, c)
            except Exception:
                parsed = parse_zone_name(zrow["name"])
                if parsed and parsed[0] in row_labels and parsed[1] in col_labels:
                    zone_pos[zrow["name"]] = (row_labels.index(parsed[0]), col_labels.index(parsed[1]))
    else:
        parsed = [parse_zone_name(z) for z in counts.keys()]
        row_labels = sorted({p[0] for p in parsed if p})
        col_labels = sorted({p[1] for p in parsed if p})

        zone_pos = {}
        for zone, p in zip(counts.keys(), parsed):
            if p and p[0] in row_labels and p[1] in col_labels:
                zone_pos[zone] = (row_labels.index(p[0]), col_labels.index(p[1]))

    if not row_labels or not col_labels:
        return np.zeros((0, 0)), [], []

    matrix = np.zeros((len(row_labels), len(col_labels)), dtype=int)

    for zone, count in counts.items():
        pos = zone_pos.get(zone)
        if pos is None:
            parsed = parse_zone_name(zone)
            if parsed and parsed[0] in row_labels and parsed[1] in col_labels:
                pos = (row_labels.index(parsed[0]), col_labels.index(parsed[1]))

        if pos:
            matrix[pos[0], pos[1]] += int(count)

    return matrix, row_labels, [str(c) for c in col_labels]


def plot_zone_heatmap(df: pd.DataFrame, title: str = "Zone heatmap"):
    """Return Plotly heatmap figure for case counts per zone."""
    matrix, row_labels, col_labels = zone_grid_data(df)
    if matrix.size == 0:
        return None

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=col_labels,
            y=row_labels,
            colorscale="Reds",
            colorbar=dict(title="Cases"),
            hoverongaps=True,
        )
    )
    fig.update_layout(
        title=title,
        height=420,
        xaxis_title="Column",
        yaxis_title="Row",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_heatmap_png(df: pd.DataFrame):
    """Create a PNG heatmap image for PDF embedding using matplotlib."""
    matrix, row_labels, col_labels = zone_grid_data(df)
    if matrix.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Cases by zone")

    max_val = matrix.max() if matrix.size else 0
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            if val > 0:
                ax.text(
                    c,
                    r,
                    str(val),
                    ha="center",
                    va="center",
                    color="white" if max_val and val > max_val / 2 else "black",
                    fontsize=8,
                )

    fig.colorbar(im, ax=ax, label="Cases")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# =============================================================================
# PDF bulletin
# =============================================================================

def register_pdf_font(lang_code: str):
    """
    Try to register a Unicode font for Urdu/Hindi PDF output.
    Falls back to Helvetica/English if not found.
    """
    if lang_code == "en":
        return "Helvetica"

    candidates = {
        "ur": [
            "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/NotoNastaliqUrdu-Regular.ttf",
        ],
        "hi": [
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf",
        ],
    }

    for path in candidates.get(lang_code, []):
        if os.path.exists(path):
            try:
                name = f"LangFont-{lang_code}"
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                continue

    return None


def build_weekly_bulletin_pdf(lang_code: str = "en") -> bytes:
    """
    Build weekly outbreak bulletin PDF:
    - KPI summary
    - top issues
    - heatmap image
    - active alerts
    - farmer advisories
    """
    df = load_reports()
    alerts = active_alerts_df()

    end = date.today()
    start = end - timedelta(days=6)

    recent = pd.DataFrame()
    if not df.empty:
        df2 = df.copy()
        df2["date_dt"] = pd.to_datetime(df2["date"], errors="coerce")
        recent = df2[
            (df2["date_dt"].dt.date >= start) &
            (df2["date_dt"].dt.date <= end)
        ]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="OutbreakWatch Weekly Bulletin",
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("OutbreakWatch Weekly Outbreak Bulletin", styles["Title"]))
    story.append(Paragraph(f"Period: {start.isoformat()} to {end.isoformat()}", styles["Normal"]))
    story.append(Spacer(1, 6))

    total_reports = len(recent)
    active_alert_count = len(alerts)
    zones_affected = int(recent["zone"].nunique()) if total_reports else 0
    avg_sev = round(float(recent["severity"].mean()), 1) if total_reports and pd.notna(recent["severity"].mean()) else 0.0

    summary_data = [
        ["Reports (7 days)", str(total_reports)],
        ["Active alerts", str(active_alert_count)],
        ["Zones affected", str(zones_affected)],
        ["Average severity", str(avg_sev)],
    ]
    summary_table = Table(summary_data)
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Top issues", styles["Heading2"]))
    if total_reports:
        top = (
            recent.groupby("issue")
            .agg(
                reports=("issue", "size"),
                zones=("zone", "nunique"),
                avg_severity=("severity", "mean"),
            )
            .sort_values(["reports", "avg_severity"], ascending=False)
            .head(5)
            .reset_index()
        )

        table_data = [["Issue", "Reports", "Zones", "Avg severity"]]
        for _, r in top.iterrows():
            table_data.append(
                [
                    str(r["issue"]),
                    str(int(r["reports"])),
                    str(int(r["zones"])),
                    f"{float(r['avg_severity']):.1f}",
                ]
            )

        top_table = Table(table_data, repeatRows=1)
        top_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(top_table)
    else:
        story.append(Paragraph("No reports in the selected week.", styles["Normal"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Heatmap", styles["Heading2"]))

    heatmap_source = recent if total_reports else df
    img_buf = make_heatmap_png(heatmap_source)
    if img_buf:
        story.append(RLImage(img_buf, width=175 * mm, height=95 * mm))
    else:
        story.append(Paragraph("No heatmap available.", styles["Normal"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Active alerts and farmer advisories", styles["Heading2"]))

    font_name = register_pdf_font(lang_code)
    effective_lang = lang_code if font_name else "en"
    advisory_style = ParagraphStyle(
        "advisory",
        parent=styles["Normal"],
        fontName=font_name or "Helvetica",
        fontSize=10,
        leading=14,
    )

    if alerts.empty:
        story.append(Paragraph("No active alerts.", styles["Normal"]))
    else:
        for alert in alerts.to_dict("records"):
            header = (
                f"<b>{str(alert.get('issue', 'Unknown'))}</b> — "
                f"zones: {str(alert.get('zones', ''))}; "
                f"rule: {str(alert.get('rule', ''))}"
            )
            story.append(Paragraph(header, styles["Normal"]))

            advisory = generate_advisory_text(alert, effective_lang)
            advisory_safe = (
                advisory.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>")
            )
            story.append(Paragraph(advisory_safe, advisory_style))
            story.append(Spacer(1, 8))

    doc.build(story)
    return buf.getvalue()


# =============================================================================
# UI helper components
# =============================================================================

def render_alert_banner():
    """Red alert banner shown when active alerts exist."""
    alerts = active_alerts_df()
    if alerts.empty:
        return

    issues = ", ".join(alerts["issue"].dropna().unique()[:3])
    st.error(
        f"🚨 OUTBREAK ALERT: {len(alerts)} active alert(s). Issue(s): {issues}. "
        "Open the Alerts page for advisories."
    )


def save_uploaded_files(files):
    """Save uploaded photos to uploads directory and return paths."""
    paths = []
    if not files:
        return paths

    for f in files:
        ext = pathlib.Path(f.name).suffix or ".jpg"
        fname = f"{uuid.uuid4().hex}{ext}"
        fpath = UPLOAD_DIR / fname
        fpath.write_bytes(f.getbuffer())
        paths.append(str(fpath))

    return paths


# =============================================================================
# Pages
# =============================================================================

def page_dashboard():
    """Dashboard with KPI cards and main charts."""
    st.subheader("Dashboard")

    df = load_reports()
    alerts = active_alerts_df()

    if df.empty:
        st.info("No reports yet. Use Demo/Data to seed demo data or add a report.")
        return

    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

    today = date.today()
    start14 = today - timedelta(days=13)
    recent = df[
        (df["date_dt"].dt.date >= start14) &
        (df["date_dt"].dt.date <= today)
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reports (14 days)", len(recent))
    c2.metric("Active alerts", len(alerts))
    c3.metric("Zones affected (14d)", int(recent["zone"].nunique()) if not recent.empty else 0)

    avg_sev = round(float(recent["severity"].mean()), 1) if not recent.empty and pd.notna(recent["severity"].mean()) else 0.0
    c4.metric("Avg severity (14d)", avg_sev)

    left, right = st.columns([3, 2])

    with left:
        fig = plot_zone_heatmap(recent, "Cases by zone (last 14 days)")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No zone heatmap data.")

    with right:
        if not recent.empty:
            issue_counts = (
                recent.groupby("issue")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(8)
            )
            st.plotly_chart(
                px.bar(
                    issue_counts,
                    x="count",
                    y="issue",
                    orientation="h",
                    title="Top issues (14d)",
                    labels={"count": "Reports", "issue": "Issue"},
                ),
                use_container_width=True,
            )
        else:
            st.info("No reports in the last 14 days.")

    if not df.empty:
        trend = df.groupby("date").size().reset_index(name="count")
        st.plotly_chart(
            px.line(
                trend,
                x="date",
                y="count",
                title="Reports over time",
                labels={"count": "Reports", "date": "Date"},
            ),
            use_container_width=True,
        )


def page_heatmap():
    """Heatmap page with date window and crop/issue filters."""
    st.subheader("Heatmap")

    df = load_reports()
    if df.empty:
        st.info("No reports available. Seed demo data or add reports.")
        return

    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

    today = date.today()

    col1, col2 = st.columns(2)
    start = col1.date_input("Start date", today - timedelta(days=13))
    end = col2.date_input("End date", today)

    if start > end:
        start, end = end, start

    crop_options = ["All"] + sorted(df["crop"].dropna().unique().tolist())
    issue_options = ["All"] + sorted(df["issue"].dropna().unique().tolist())

    col3, col4 = st.columns(2)
    crop = col3.selectbox("Crop", crop_options)
    issue = col4.selectbox("Issue", issue_options)

    filtered = df[
        (df["date_dt"].dt.date >= start) &
        (df["date_dt"].dt.date <= end)
    ]

    if crop != "All":
        filtered = filtered[filtered["crop"] == crop]
    if issue != "All":
        filtered = filtered[filtered["issue"] == issue]

    st.caption(f"{len(filtered)} reports in selected window.")

    fig = plot_zone_heatmap(filtered, f"Cases from {start.isoformat()} to {end.isoformat()}")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cases for selected filters.")

    with st.expander("Filtered report table"):
        st.dataframe(filtered.drop(columns=["raw_json"], errors="ignore"), use_container_width=True)


def page_timeline():
    """Spread timeline replay using date slider."""
    st.subheader("Spread Timeline")

    df = load_reports()
    if df.empty:
        st.info("No reports available. Seed demo data or add reports.")
        return

    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"])

    if df.empty:
        st.info("No valid report dates available.")
        return

    min_d = df["date_dt"].dt.date.min()
    max_d = df["date_dt"].dt.date.max()

    selected = st.slider("Replay date", min_value=min_d, max_value=max_d, value=max_d)

    issue_options = ["All"] + sorted(df["issue"].dropna().unique().tolist())
    issue = st.selectbox("Issue", issue_options)

    cumulative = st.checkbox("Cumulative cases up to selected date", value=True)

    view = df.copy()
    if issue != "All":
        view = view[view["issue"] == issue]

    if cumulative:
        view = view[view["date_dt"].dt.date <= selected]
        title = f"Cumulative cases up to {selected.isoformat()}"
    else:
        view = view[view["date_dt"].dt.date == selected]
        title = f"New cases on {selected.isoformat()}"

    fig = plot_zone_heatmap(view, title)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cases for selected timeline settings.")

    if cumulative:
        prev_date = selected - timedelta(days=1)
        prev = df[df["date_dt"].dt.date <= prev_date]
        if issue != "All":
            prev = prev[prev["issue"] == issue]

        curr_counts = view.groupby("zone").size()
        prev_counts = prev.groupby("zone").size()
        new_cases = (curr_counts.subtract(prev_counts, fill_value=0)).astype(int)
        new_cases = new_cases[new_cases > 0].reset_index(name="new_cases_since_previous_day")

        if not new_cases.empty:
            st.write("Zone movement / new detections since previous day")
            st.dataframe(new_cases, use_container_width=True)
        else:
            st.caption("No new zone detections compared with previous day.")
    else:
        day_cases = view.groupby("zone").size().reset_index(name="cases")
        st.dataframe(day_cases, use_container_width=True)


def page_new_report():
    """Manual/photo report entry page."""
    st.subheader("New Report")

    zones = zone_names()
    if not zones:
        st.warning("No zones defined. Creating default A1-D6 grid.")
        save_zones(4, 6)
        zones = zone_names()

    with st.form("new_report", clear_on_submit=False):
        crop_choice = st.selectbox("Crop", COMMON_CROPS)

        if crop_choice == "Other":
            crop = st.text_input("Enter crop name")
        else:
            crop = crop_choice

        zone = st.selectbox("Zone", zones)
        report_date = st.date_input("Observation date", date.today())
        reporter = st.text_input("Reporter name (optional)")
        notes = st.text_area(
            "Symptoms / notes (optional)",
            placeholder="Example: curled leaves, yellowing, aphids under leaf, wilting...",
        )
        photos = st.file_uploader(
            "Photo(s)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )

        submitted = st.form_submit_button("Save and analyze")

    if submitted:
        if not crop or not str(crop).strip():
            st.error("Crop is required.")
            return

        if not zone:
            st.error("Zone is required.")
            return

        paths = save_uploaded_files(photos) if photos else []
        first_bytes = photos[0].getvalue() if photos else None

        with st.spinner("Analyzing report..."):
            analysis = analyze_with_ai(first_bytes, str(crop).strip(), zone, (notes or "").strip())

        record = {
            "photo_path": ";".join(paths) if paths else None,
            "crop": str(crop).strip(),
            "zone": zone,
            "date": report_date.isoformat(),
            "reporter": reporter.strip() if reporter else None,
            "issue": analysis.get("issue_name"),
            "issue_type": analysis.get("issue_type"),
            "severity": analysis.get("severity"),
            "confidence": analysis.get("confidence"),
            "visible_symptoms": json.dumps(analysis.get("visible_symptoms", []), ensure_ascii=False),
            "likely_spread_mode": analysis.get("likely_spread_mode"),
            "first_actions_non_chemical": json.dumps(analysis.get("first_actions_non_chemical", []), ensure_ascii=False),
            "first_actions_chemical": json.dumps(analysis.get("first_actions_chemical", []), ensure_ascii=False),
            "raw_json": json.dumps(analysis, ensure_ascii=False),
        }

        report_id = insert_report(record)
        new_alerts = run_outbreak_rules()

        st.success(f"Report #{report_id} saved.")

        if new_alerts:
            st.error(f"{new_alerts} new outbreak alert condition(s) detected.")

        with st.expander("AI / demo analysis result", expanded=True):
            st.json(analysis)


def page_alerts():
    """Alerts page with advisory generation and acknowledge buttons."""
    st.subheader("Outbreak Alerts")

    if st.button("Re-run outbreak rules"):
        n = run_outbreak_rules()
        st.success(f"Rule engine finished. {n} new alert(s) inserted.")
        st.rerun()

    alerts = all_alerts_df()
    if alerts.empty:
        st.success("No alerts found.")
        return

    lang = st.session_state.get("lang_code", "en")

    for alert in alerts.to_dict("records"):
        status = "Active" if int(alert.get("acknowledged", 0)) == 0 else "Acknowledged"
        icon = "🚨" if status == "Active" else "✅"

        with st.container():
            st.markdown(f"#### {icon} {alert.get('issue', 'Unknown')}")
            st.caption(
                f"Rule: {alert.get('rule', '')} | Zones: {alert.get('zones', '')} | "
                f"Created: {alert.get('created_at', '')} | Status: {status}"
            )

            colA, colB = st.columns([1, 2])

            if int(alert.get("acknowledged", 0)) == 0:
                if colA.button("Acknowledge", key=f"ack_{alert.get('id')}"):
                    acknowledge_alert(alert.get("id"))
                    st.rerun()

            with st.expander("Farmer advisory", expanded=False):
                if st.button("Generate / refresh advisory", key=f"adv_{alert.get('id')}_{lang}"):
                    st.session_state[f"show_adv_{alert.get('id')}_{lang}"] = True

                if st.session_state.get(f"show_adv_{alert.get('id')}_{lang}", False):
                    advisory = generate_advisory_text(alert, lang)
                    st.write(advisory)

            st.divider()


def page_zone_setup():
    """Zone grid setup page."""
    st.subheader("Zone Setup")

    zones_df = load_zones()
    st.write(f"Current zones: {len(zones_df)}")

    if not zones_df.empty:
        st.dataframe(zones_df, use_container_width=True)

    with st.form("zone_form"):
        rows = st.number_input("Number of rows (A, B, C...)", min_value=1, max_value=26, value=4)
        cols = st.number_input("Number of columns", min_value=1, max_value=20, value=6)
        custom = st.text_area(
            "Optional custom zone names, comma separated, row-major. Leave blank for automatic A1, A2, ... names.",
            placeholder="A1,A2,A3,A4,A5,A6,B1,B2,...",
            height=90,
        )

        submitted = st.form_submit_button("Save zone grid")
        if submitted:
            save_zones(rows, cols, custom)
            st.success("Zone grid saved.")
            st.rerun()

    st.caption("Changing zones replaces the zone list but keeps historical reports intact.")


def page_bulletin():
    """Weekly PDF bulletin page."""
    st.subheader("Weekly Outbreak Bulletin (PDF)")

    lang = st.session_state.get("lang_code", "en")
    st.caption("Includes last 7 days top issues, heatmap image, active alerts, and farmer advisories.")

    if st.button("Build bulletin PDF"):
        with st.spinner("Building PDF..."):
            try:
                pdf_bytes = build_weekly_bulletin_pdf(lang)
                st.download_button(
                    "Download bulletin PDF",
                    data=pdf_bytes,
                    file_name=f"outbreak_bulletin_{date.today().isoformat()}.pdf",
                    mime="application/pdf",
                )
            except Exception as exc:
                st.error(f"Could not build PDF: {exc}")


def page_demo():
    """Demo/data tools page."""
    st.subheader("Demo Data and Tools")

    st.info("This app automatically loads realistic demo data on first launch so you can test immediately.")

    col1, col2, col3 = st.columns(3)

    if col1.button("Seed / reset demo data"):
        seed_demo_data(force=True)
        st.success("Demo data loaded.")
        st.rerun()

    if col2.button("Re-run outbreak engine"):
        n = run_outbreak_rules()
        st.success(f"Done. {n} new alert(s) inserted.")

    if col3.button("Clear reports & alerts"):
        clear_all_data()
        st.warning("Reports and alerts cleared.")
        st.rerun()

    df = load_reports()
    st.write("Latest reports")
    st.dataframe(df.head(100).drop(columns=["raw_json"], errors="ignore"), use_container_width=True)


# =============================================================================
# Main app
# =============================================================================

def main():
    """Main Streamlit entry point."""
    init_db()

    # Ensure default zones exist
    if load_zones().empty:
        save_zones(4, 6)

    # Auto-load demo data on first run so dashboard is never blank
    if load_reports().empty:
        seed_demo_data(force=False)

    with st.sidebar:
        st.title("🌿 OutbreakWatch")
        st.caption("Pest & disease cluster monitoring")

        current_code = st.session_state.get("lang_code", "en")
        current_name = CODE_LANG.get(current_code, "English")
        idx = LANG_OPTIONS.index(current_name) if current_name in LANG_OPTIONS else 0

        lang_name = st.selectbox("AI advisory language", LANG_OPTIONS, index=idx)
        st.session_state["lang_code"] = LANG_CODE[lang_name]

        menu = st.radio(
            "Menu",
            [
                "Dashboard",
                "Heatmap",
                "Spread Timeline",
                "New Report",
                "Alerts",
                "Zone Setup",
                "Weekly Bulletin",
                "Demo/Data",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        if gemini_available():
            st.success("Gemini API connected.")
        else:
            st.info("No Gemini API key found. Running in offline demo AI mode. Add GEMINI_API_KEY to enable real Gemini vision.")

        st.caption(f"SQLite DB: {DB_PATH.name}")

    render_alert_banner()

    if menu == "Dashboard":
        page_dashboard()
    elif menu == "Heatmap":
        page_heatmap()
    elif menu == "Spread Timeline":
        page_timeline()
    elif menu == "New Report":
        page_new_report()
    elif menu == "Alerts":
        page_alerts()
    elif menu == "Zone Setup":
        page_zone_setup()
    elif menu == "Weekly Bulletin":
        page_bulletin()
    elif menu == "Demo/Data":
        page_demo()


# Run app
main()