import sqlite3
import bcrypt
import hashlib

DB_NAME = 'lcl_users.db'

# Free trial code — publicly shared, never marked as used
FREE_TRIAL_CODE = 'FREETRIAL2025'


def init_db():
    """Initialize database and insert the free trial code if not exists."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT,
            name TEXT,
            password TEXT,
            session_token TEXT
        )
    ''')

    # Activation codes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activation_codes (
            code TEXT PRIMARY KEY,
            is_used INTEGER DEFAULT 0
        )
    ''')

    # Feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_type TEXT NOT NULL,
            content TEXT NOT NULL,
            username_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Auto-insert the free trial code (safe to run multiple times)
    cursor.execute(
        "INSERT OR IGNORE INTO activation_codes (code, is_used) VALUES (?, 0)",
        (FREE_TRIAL_CODE,)
    )

    conn.commit()
    conn.close()


def get_all_users_for_auth():
    """Return all users in the format required by streamlit-authenticator."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT username, email, name, password FROM users")
    users_data = cursor.fetchall()
    conn.close()

    credentials = {"usernames": {}}
    for username, email, name, password in users_data:
        credentials["usernames"][username] = {
            "email": email,
            "name": name,
            "password": password
        }
    return credentials


def register_new_user(username, email, name, plain_password, activation_code):
    """Register a new user with activation code verification."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Check activation code exists
    cursor.execute("SELECT is_used FROM activation_codes WHERE code = ?", (activation_code,))
    result = cursor.fetchone()

    if result is None:
        conn.close()
        return False, "❌ Invalid access code. Please check and try again."

    # Free trial code is never marked as used — skip the used check
    if activation_code != FREE_TRIAL_CODE and result[0] == 1:
        conn.close()
        return False, "❌ This access code has already been used."

    # Check username not taken
    cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
    if cursor.fetchone() is not None:
        conn.close()
        return False, "❌ This username is already taken. Please choose another."

    hashed_password = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    try:
        cursor.execute(
            "INSERT INTO users (username, email, name, password) VALUES (?, ?, ?, ?)",
            (username, email, name, hashed_password)
        )
        # Only mark as used if it's NOT the free trial code
        if activation_code != FREE_TRIAL_CODE:
            cursor.execute(
                "UPDATE activation_codes SET is_used = 1 WHERE code = ?",
                (activation_code,)
            )
        conn.commit()
        conn.close()
        return True, "✅ Account created! Please log in with your new credentials."
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, f"❌ Registration error: {str(e)}"


# ── Session token (multi-device lock) ────────────────────

def update_session_token(username, token):
    """Update the latest session token for a user."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET session_token = ? WHERE username = ?",
        (token, username)
    )
    conn.commit()
    conn.close()


def get_session_token(username):
    """Get the stored session token for a user."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT session_token FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


# ── Feedback module ───────────────────────────────────────

def init_feedback_table():
    """No-op — feedback table is initialized in init_db()."""
    pass


def submit_feedback(username: str, feedback_type: str, content: str):
    """Submit a feedback entry (anonymized display name)."""
    salt = "powertool_feedback_2024"
    hash_str = hashlib.sha256(f"{username}{salt}".encode()).hexdigest()[:6].upper()
    display_name = f"Engineer #{hash_str}"

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        '''INSERT INTO feedback (feedback_type, content, username_hash, display_name, created_at)
           VALUES (?, ?, ?, ?, datetime('now', 'localtime'))''',
        (feedback_type, content, hash_str, display_name)
    )
    conn.commit()
    conn.close()


def get_feedback(feedback_type: str):
    """Return feedback entries of a given type, newest first."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        '''SELECT display_name, content, created_at
           FROM feedback
           WHERE feedback_type = ?
           ORDER BY created_at DESC''',
        (feedback_type,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"display_name": r[0], "content": r[1], "created_at": r[2]} for r in rows]


if __name__ == "__main__":
    init_db()
    print("✅ Database initialized. Free trial code inserted:", FREE_TRIAL_CODE)
