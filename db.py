"""SQLite persistence layer: training data, model registry, patient assessments."""
import sqlite3, csv, os
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, "diabetes.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Pregnancies REAL, Glucose REAL, BloodPressure REAL, SkinThickness REAL,
    Insulin REAL, BMI REAL, DiabetesPedigreeFunction REAL, Age REAL, Outcome INTEGER
);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    model TEXT, accuracy REAL, roc_auc REAL, f1 REAL, recall REAL, precision REAL,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    patient_name TEXT, age REAL, pregnancies REAL, glucose REAL,
    blood_pressure REAL, bmi REAL, family_index REAL,
    quantum_prob REAL, classical_prob REAL, risk_level TEXT, notes TEXT
);
"""

@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()

def init_db():
    with conn() as c:
        c.executescript(SCHEMA)
        if c.execute("SELECT COUNT(*) FROM dataset").fetchone()[0] == 0:
            path = os.path.join(ROOT, "data", "diabetes.csv")
            with open(path, newline="") as f:
                r = csv.DictReader(f)
                rows = [tuple(float(row[k]) if k != "Outcome" else int(row[k]) for k in r.fieldnames) for row in r]
            c.executemany("INSERT INTO dataset (Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome) VALUES (?,?,?,?,?,?,?,?,?)", rows)

def add_assessment(d):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO assessments (patient_name,age,pregnancies,glucose,blood_pressure,bmi,family_index,quantum_prob,classical_prob,risk_level,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (d["patient_name"], d["age"], d["pregnancies"], d["glucose"], d["blood_pressure"], d["bmi"],
             d["family_index"], d["quantum_prob"], d["classical_prob"], d["risk_level"], d.get("notes", "")))
        return cur.lastrowid

def list_assessments():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM assessments ORDER BY id DESC")]

def delete_assessment(i):
    with conn() as c:
        c.execute("DELETE FROM assessments WHERE id=?", (i,))

def save_run(model, m, notes=""):
    with conn() as c:
        c.execute("INSERT INTO training_runs (model,accuracy,roc_auc,f1,recall,precision,notes) VALUES (?,?,?,?,?,?,?)",
                  (model, m["accuracy"], m["roc_auc"], m["f1"], m["recall"], m["precision"], notes))

def load_dataset():
    import pandas as pd
    with conn() as c:
        return pd.read_sql_query("SELECT Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome FROM dataset", c)
