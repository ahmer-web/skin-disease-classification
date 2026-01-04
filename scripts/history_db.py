import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("history.db")


def init_db():
    """Create predictions table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            username TEXT,
            filename TEXT,
            mode TEXT,              -- 'single' or 'batch'
            predicted_class TEXT,
            confidence REAL,
            risk_level TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_prediction(username, filename, mode, predicted_class, confidence, risk_level):
    """Insert one prediction record."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions
        (timestamp, username, filename, mode, predicted_class, confidence, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            username,
            filename,
            mode,
            predicted_class,
            float(confidence),
            risk_level,
        ),
    )
    conn.commit()
    conn.close()


def get_predictions_for_user(username, limit=200):
    """Get recent predictions for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, timestamp, username, filename, mode,
               predicted_class, confidence, risk_level
        FROM predictions
        WHERE username=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (username, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_predictions(limit=200):
    """Get all predictions (for admin)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, timestamp, username, filename, mode,
               predicted_class, confidence, risk_level
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_prediction(record_id: int):
    """Delete a single prediction row by ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id=?", (record_id,))
    conn.commit()
    conn.close()


def clear_history_for_user(username: str):
    """Delete all predictions for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE username=?", (username,))
    conn.commit()
    conn.close()


def clear_all_history():
    """Delete all prediction records."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
