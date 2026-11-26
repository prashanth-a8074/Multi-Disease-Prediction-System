import os
import json
import sqlite3
import hashlib
import mysql.connector
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Fallback config
USE_SQLITE = os.getenv("USE_SQLITE_FALLBACK", "True").lower() == "true"

def get_connection():
    """Establish MySQL connection, falling back to SQLite if requested or MySQL fails."""
    if USE_SQLITE:
        return sqlite3.connect("multi_disease_db.sqlite")
    
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "your_password"),
            database=os.getenv("DB_NAME", "multi_disease_db"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        return conn
    except Exception as e:
        print(f"MySQL connection failed: {e}. Falling back to SQLite local database...")
        # Automatically toggle to SQLite
        return sqlite3.connect("multi_disease_db.sqlite")

def hash_password(password):
    """Hash password securely using SHA-256 (standard built-in, zero-dependency)."""
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    """Create database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    is_sqlite = isinstance(conn, sqlite3.Connection)
    
    # Create tables based on database type
    if is_sqlite:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """.replace("INTEGER PRIMARY KEY AUTO_INCREMENT", "INTEGER PRIMARY KEY AUTOINCREMENT"))
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            patient_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            disease TEXT NOT NULL,
            risk_percentage REAL NOT NULL,
            prediction_result INTEGER NOT NULL,
            feature_values TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """)
    else:
        # MySQL tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            email VARCHAR(150) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            patient_name VARCHAR(150) NOT NULL,
            age INT NOT NULL,
            gender VARCHAR(20) NOT NULL,
            disease VARCHAR(100) NOT NULL,
            risk_percentage FLOAT NOT NULL,
            prediction_result INT NOT NULL,
            feature_values TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """)
    
    conn.commit()
    conn.close()
    print("Database tables initialized successfully.")

# Initialize database right away
init_db()

# ---------------------------------------------------------
# DATABASE API FUNCTIONS
# ---------------------------------------------------------

def register_user(username, email, password):
    """Registers a new user in the database. Returns True if success, False otherwise."""
    conn = get_connection()
    cursor = conn.cursor()
    pwd_hash = hash_password(password)
    
    is_sqlite = isinstance(conn, sqlite3.Connection)
    placeholder = "?" if is_sqlite else "%s"
    
    try:
        query = f"INSERT INTO users (username, email, password_hash) VALUES ({placeholder}, {placeholder}, {placeholder})"
        cursor.execute(query, (username, email, pwd_hash))
        conn.commit()
        success = True
    except Exception as e:
        print(f"Error registering user: {e}")
        success = False
    finally:
        conn.close()
    return success

def authenticate_user(username, password):
    """Verifies credentials. Returns user dict (id, username, email) if successful, else None."""
    conn = get_connection()
    cursor = conn.cursor()
    pwd_hash = hash_password(password)
    
    is_sqlite = isinstance(conn, sqlite3.Connection)
    placeholder = "?" if is_sqlite else "%s"
    
    query = f"SELECT id, username, email FROM users WHERE username={placeholder} AND password_hash={placeholder}"
    cursor.execute(query, (username, pwd_hash))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {'id': row[0], 'username': row[1], 'email': row[2]}
    return None

def save_prediction(user_id, patient_name, age, gender, disease, risk_percentage, prediction_result, feature_values):
    """Saves a prediction entry to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    is_sqlite = isinstance(conn, sqlite3.Connection)
    p = "?" if is_sqlite else "%s"
    
    # Serialize feature values if dictionary
    if isinstance(feature_values, dict):
        feature_values = json.dumps(feature_values)
        
    try:
        query = f"""
        INSERT INTO predictions 
        (user_id, patient_name, age, gender, disease, risk_percentage, prediction_result, feature_values) 
        VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
        """
        cursor.execute(query, (user_id, patient_name, age, gender, disease, float(risk_percentage), int(prediction_result), feature_values))
        conn.commit()
        success = True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        success = False
    finally:
        conn.close()
    return success

def get_user_history(user_id):
    """Retrieves prediction logs for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    
    is_sqlite = isinstance(conn, sqlite3.Connection)
    p = "?" if is_sqlite else "%s"
    
    query = f"""
    SELECT id, patient_name, age, gender, disease, risk_percentage, prediction_result, feature_values, created_at 
    FROM predictions 
    WHERE user_id={p} 
    ORDER BY created_at DESC
    """
    cursor.execute(query, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            'id': r[0],
            'patient_name': r[1],
            'age': r[2],
            'gender': r[3],
            'disease': r[4],
            'risk_percentage': r[5],
            'prediction_result': r[6],
            'feature_values': json.loads(r[7]),
            'created_at': r[8]
        })
    return history

def get_all_predictions():
    """Retrieves all predictions for global statistics in the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT id, patient_name, age, gender, disease, risk_percentage, prediction_result, created_at 
    FROM predictions 
    ORDER BY created_at DESC
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    predictions = []
    for r in rows:
        predictions.append({
            'id': r[0],
            'patient_name': r[1],
            'age': r[2],
            'gender': r[3],
            'disease': r[4],
            'risk_percentage': r[5],
            'prediction_result': r[6],
            'created_at': r[7]
        })
    return predictions
