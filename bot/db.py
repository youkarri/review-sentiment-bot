import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "reviews.db"


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.commit()


def save_reviews(user_id, rows):
    """rows: список кортежей (текст, класс, уверенность)."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.executemany(
            "INSERT INTO reviews (user_id, text, label, confidence) VALUES (?, ?, ?, ?)",
            [(user_id, t, label, conf) for t, label, conf in rows],
        )
        conn.commit()


def get_stats(user_id):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute(
            "SELECT label, COUNT(*) FROM reviews WHERE user_id = ? GROUP BY label",
            (user_id,),
        )
        return Counter(dict(cur.fetchall()))