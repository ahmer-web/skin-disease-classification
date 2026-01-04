import sqlite3
import hashlib
import streamlit as st
from pathlib import Path

DB_PATH = Path("users.db")


def init_db():
    """Create users table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            is_admin INTEGER DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username: str, password: str):
    """
    Register a new user.
    First ever user becomes admin (is_admin=1).
    Returns (success: bool, is_admin: bool, error_message: str|None)
    """
    if not username or not password:
        return False, False, "Username and password are required."

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Check if there are any users already
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        is_admin = 1 if count == 0 else 0

        cursor.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
            (username, hash_password(password), is_admin),
        )
        conn.commit()
        return True, bool(is_admin), None
    except sqlite3.IntegrityError:
        return False, False, "Username already exists."
    finally:
        conn.close()


def login_user(username: str, password: str):
    """
    Validate credentials.
    Returns (success: bool, is_admin: bool)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT is_admin FROM users WHERE username=? AND password=?",
        (username, hash_password(password)),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return False, False
    return True, bool(row[0])


def login_session(username: str, is_admin: bool):
    st.session_state["logged_in"] = True
    st.session_state["username"] = username
    st.session_state["is_admin"] = is_admin


def logout_session():
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["is_admin"] = False
