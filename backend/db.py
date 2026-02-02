from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "backend" / "data" / "app.db"
DDL_PATH = ROOT_DIR / "SQL_DDL.sql"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    ddl_sql = DDL_PATH.read_text(encoding="utf-8")
    with get_conn() as conn:
        conn.executescript(ddl_sql)
        cols = conn.execute("PRAGMA table_info(all_seed_info)").fetchall()
        if cols:
            col_names = {row["name"] for row in cols}
            if "created_at" not in col_names:
                conn.execute("ALTER TABLE all_seed_info ADD COLUMN created_at TEXT")
            if "updated_at" not in col_names:
                conn.execute("ALTER TABLE all_seed_info ADD COLUMN updated_at TEXT")

        utterance_cols = conn.execute("PRAGMA table_info(utterance)").fetchall()
        if utterance_cols:
            utterance_names = {row["name"] for row in utterance_cols}
            if "did_asked_knowledge" not in utterance_names:
                conn.execute("ALTER TABLE utterance ADD COLUMN did_asked_knowledge INTEGER NOT NULL DEFAULT 0")

        layout_runs_cols = conn.execute("PRAGMA table_info(layout_runs)").fetchall()
        if layout_runs_cols:
            layout_runs_names = {row["name"] for row in layout_runs_cols}
            if "is_active" not in layout_runs_names:
                conn.execute("ALTER TABLE layout_runs ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
            if "params_json" not in layout_runs_names:
                conn.execute("ALTER TABLE layout_runs ADD COLUMN params_json TEXT")
            if "created_at" not in layout_runs_names:
                conn.execute("ALTER TABLE layout_runs ADD COLUMN created_at TEXT")

        layout_points_cols = conn.execute("PRAGMA table_info(layout_points)").fetchall()
        if layout_points_cols:
            layout_points_names = {row["name"] for row in layout_points_cols}
            if "is_active" not in layout_points_names:
                conn.execute("ALTER TABLE layout_points ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
            if "created_at" not in layout_points_names:
                conn.execute("ALTER TABLE layout_points ADD COLUMN created_at TEXT")
