#!/usr/bin/env python3
"""
Analytics Module - Track application usage and health
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Setup analytics database
ANALYTICS_DB = Path("data/analytics/analytics.db")
ANALYTICS_DB.parent.mkdir(parents=True, exist_ok=True)

class Analytics:
    def __init__(self, db_path=ANALYTICS_DB):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT
            )
        """)
        
        # App events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                event_type TEXT,
                timestamp TIMESTAMP,
                details TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Classification results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP,
                predicted_type INTEGER,
                confidence REAL,
                user_corrected BOOLEAN,
                correct_type INTEGER,
                processing_time_ms REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Errors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP,
                error_type TEXT,
                error_message TEXT,
                traceback TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Health checks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                status TEXT,
                model_loaded BOOLEAN,
                database_ok BOOLEAN,
                disk_space_mb REAL,
                response_time_ms REAL,
                details TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_id, ip_address="", user_agent=""):
        """Create new user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (session_id, start_time, ip_address, user_agent)
            VALUES (?, ?, ?, ?)
        """, (session_id, datetime.now(), ip_address, user_agent))
        
        conn.commit()
        conn.close()
    
    def log_event(self, session_id, event_type, details=""):
        """Log an event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO events (session_id, event_type, timestamp, details)
            VALUES (?, ?, ?, ?)
        """, (session_id, event_type, datetime.now(), details))
        
        conn.commit()
        conn.close()
    
    def log_classification(self, session_id, predicted_type, confidence, 
                          user_corrected=False, correct_type=None, processing_time_ms=0):
        """Log classification result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO classifications 
            (session_id, timestamp, predicted_type, confidence, user_corrected, correct_type, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, datetime.now(), predicted_type, confidence, user_corrected, correct_type, processing_time_ms))
        
        conn.commit()
        conn.close()
    
    def log_error(self, session_id, error_type, error_message, traceback=""):
        """Log an error"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO errors (session_id, timestamp, error_type, error_message, traceback)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, datetime.now(), error_type, error_message, traceback))
        
        conn.commit()
        conn.close()
    
    def log_health_check(self, status, model_loaded, database_ok, disk_space_mb, response_time_ms, details=""):
        """Log health check"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO health_checks (timestamp, status, model_loaded, database_ok, disk_space_mb, response_time_ms, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now(), status, model_loaded, database_ok, disk_space_mb, response_time_ms, details))
        
        conn.commit()
        conn.close()
    
    def get_sessions_count(self, days=7):
        """Get number of sessions in last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cursor.execute("""
            SELECT COUNT(*) FROM sessions WHERE start_time > ?
        """, (cutoff_date,))
        
        result = cursor.fetchone()[0]
        conn.close()
        return result
    
    def get_classifications_count(self, days=7):
        """Get number of classifications in last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cursor.execute("""
            SELECT COUNT(*) FROM classifications WHERE timestamp > ?
        """, (cutoff_date,))
        
        result = cursor.fetchone()[0]
        conn.close()
        return result
    
    def get_errors_count(self, days=7):
        """Get number of errors in last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cursor.execute("""
            SELECT COUNT(*) FROM errors WHERE timestamp > ?
        """, (cutoff_date,))
        
        result = cursor.fetchone()[0]
        conn.close()
        return result
    
    def get_accuracy_stats(self, days=7):
        """Get accuracy statistics"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get classifications where user corrected
        df = pd.read_sql_query("""
            SELECT predicted_type, correct_type, confidence
            FROM classifications
            WHERE timestamp > ? AND user_corrected = 1
        """, conn, params=(cutoff_date,))
        
        conn.close()
        
        if len(df) == 0:
            return {"total": 0, "correct": 0, "accuracy": 0}
        
        correct = (df['predicted_type'] == df['correct_type']).sum()
        accuracy = 100 * correct / len(df)
        
        return {
            "total": len(df),
            "correct": correct,
            "accuracy": accuracy
        }
    
    def get_daily_stats(self, days=30):
        """Get daily statistics"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        df = pd.read_sql_query("""
            SELECT DATE(start_time) as date, COUNT(*) as sessions
            FROM sessions
            WHERE start_time > ?
            GROUP BY DATE(start_time)
            ORDER BY date DESC
        """, conn, params=(cutoff_date,))
        
        conn.close()
        return df
    
    def get_classification_type_distribution(self, days=7):
        """Get distribution of classification types"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        df = pd.read_sql_query("""
            SELECT predicted_type, COUNT(*) as count
            FROM classifications
            WHERE timestamp > ?
            GROUP BY predicted_type
            ORDER BY predicted_type
        """, conn, params=(cutoff_date,))
        
        conn.close()
        
        # Fill missing types
        for i in range(1, 8):
            if i not in df['predicted_type'].values:
                df = pd.concat([df, pd.DataFrame({'predicted_type': [i], 'count': [0]})], ignore_index=True)
        
        return df.sort_values('predicted_type')
    
    def get_latest_health_check(self):
        """Get latest health check"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, status, model_loaded, database_ok, disk_space_mb, response_time_ms
            FROM health_checks
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "timestamp": result[0],
                "status": result[1],
                "model_loaded": bool(result[2]),
                "database_ok": bool(result[3]),
                "disk_space_mb": result[4],
                "response_time_ms": result[5]
            }
        return None
    
    def get_errors_by_type(self, days=7):
        """Get errors grouped by type"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        df = pd.read_sql_query("""
            SELECT error_type, COUNT(*) as count
            FROM errors
            WHERE timestamp > ?
            GROUP BY error_type
            ORDER BY count DESC
        """, conn, params=(cutoff_date,))
        
        conn.close()
        return df

# Create global analytics instance
analytics = Analytics()
