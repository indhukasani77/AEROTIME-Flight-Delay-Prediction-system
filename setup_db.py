"""
AEROTIME — SQLite Database Setup
Creates 3 tables: users, predictions, flights
Stores user information and all flight details
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aerotime.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets you access columns by name
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # ── USERS TABLE ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name               TEXT NOT NULL,
            last_name                TEXT NOT NULL,
            email                    TEXT UNIQUE NOT NULL,
            username                 TEXT UNIQUE NOT NULL,
            password                 TEXT NOT NULL,
            organization             TEXT,
            preferred_airline        TEXT,
            preferred_dep_airport    TEXT,
            preferred_arr_airport    TEXT,
            preferred_aircraft_type  TEXT,
            created_at               DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ── PREDICTIONS TABLE ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER,
            dep_airport      TEXT,
            arr_airport      TEXT,
            airline          TEXT,
            aircraft_type    TEXT,
            flight_date      TEXT,
            dep_hour         INTEGER,
            weather_severity INTEGER,
            congestion       INTEGER,
            wind_speed       REAL,
            visibility       REAL,
            delay_minutes    INTEGER,
            confidence       INTEGER,
            probability      REAL,
            model_used       TEXT,
            created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # ── FLIGHTS TABLE ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flights (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER,
            flight_number  TEXT,
            route          TEXT,
            flight_date    TEXT,
            dep_time       TEXT,
            airline        TEXT,
            aircraft       TEXT,
            delay_minutes  INTEGER,
            status         TEXT,
            uploaded_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")
    print("📊 Tables: users, predictions, flights")
    print(f"📁 Location: {DB_PATH}")


if __name__ == '__main__':
    init_db()