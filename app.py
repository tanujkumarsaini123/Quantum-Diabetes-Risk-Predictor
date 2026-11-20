"""Flask server: quantum diabetes-risk API + web UI, persisted in SQLite."""
import csv, io, json, os, threading, webbrowser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from flask import Flask, jsonify, render_template, request, Response
import db
from qml_core import Preprocessor, QuantumClassifier, FEATURES

ROOT = db.ROOT
MODEL_DIR = os.path.join(ROOT, "model")
app = Flask(__name__)
db.init_db()

if not os.path.exists(os.path.join(MODEL_DIR, "qml_weights.json")):
    print("No trained model found - training one now (about 2 minutes)...")
    import train
    os.makedirs(MODEL_DIR, exist_ok=True)
    train.train(epochs=100, layers=4)

_w = json.load(open(os.path.join(MODEL_DIR, "qml_weights.json")))
PRE = Preprocessor(**_w["pre"])
QC = QuantumClassifier(_w["w"], _w["a"], _w["b"], _w["layers"], _w.get("s"))
_df = db.load_dataset()
_tr, _ = train_test_split(_df, test_size=0.2, stratify=_df.Outcome, random_state=42)  # same split as train.py
CLASSICAL = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500)).fit(_tr[FEATURES].values, _tr.Outcome.values)
METRICS = json.load(open(os.path.join(MODEL_DIR, "metrics.json")))
with db.conn() as _c:  # seed the model registry table on first run
    if _c.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0] == 0:
        for _n, _m in METRICS["results"].items():
            db.save_run(_n, _m, f"{METRICS['epochs']} epochs" if _n.startswith("Quantum") else "")
REF = np.median(PRE.transform(db.load_dataset()), axis=0)  # population-median patient (scaled space)

LABELS = {"Glucose": "Blood glucose", "BMI": "Body mass index", "Age": "Age",
          "BloodPressure": "Blood pressure", "DiabetesPedigreeFunction": "Family history", "Pregnancies": "Pregnancies"}
RANGES = {"age": (18, 90), "pregnancies": (0, 17), "glucose": (40, 300), "blood_pressure": (30, 140),
          "bmi": (12, 65), "family_index": (0.05, 2.5)}

def level(p):
    return "Low" if p < 0.35 else "Moderate" if p < 0.60 else "High"

def parse(d):
    out = {}
    for k, (lo, hi) in RANGES.items():
        try:
            v = float(d[k])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Enter a number for {k.replace('_', ' ')}.")
        if not lo <= v <= hi:
            raise ValueError(f"{k.replace('_', ' ').title()} should be between {lo} and {hi}.")
        out[k] = v
    return out

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/predict")
def predict():
    d = request.get_json(force=True)
    try:
        v = parse(d)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    row = pd.DataFrame([{"Glucose": v["glucose"], "BMI": v["bmi"], "Age": v["age"], "BloodPressure": v["blood_pressure"],
                         "DiabetesPedigreeFunction": v["family_index"], "Pregnancies": v["pregnancies"]}])
    x = PRE.transform(row)[0]
    # one batched circuit call: the patient + six "feature set to population median" variants
    variants = np.vstack([x] + [np.where(np.arange(len(x)) == i, REF[i], x) for i in range(len(x))])
    probs = QC.proba(variants)
    q = float(probs[0])
    contrib = [{"feature": LABELS[f], "impact": float(q - probs[i + 1])} for i, f in enumerate(FEATURES)]
    contrib.sort(key=lambda c: -abs(c["impact"]))
    c = float(CLASSICAL.predict_proba(row[FEATURES].values)[0, 1])
    rec = {"patient_name": (d.get("patient_name") or "Anonymous").strip()[:60] or "Anonymous", **v,
           "quantum_prob": q, "classical_prob": c, "risk_level": level(q), "notes": ""}
    rec["id"] = db.add_assessment(rec)
    return jsonify(id=rec["id"], quantum_prob=q, classical_prob=c, risk_level=rec["risk_level"], contributions=contrib,
                   agree=level(q) == level(c))

@app.get("/api/history")
def history():
    rows = db.list_assessments()
    n = len(rows)
    return jsonify(rows=rows, stats={"total": n, "high": sum(r["risk_level"] == "High" for r in rows),
                                     "avg": (sum(r["quantum_prob"] for r in rows) / n) if n else 0})

@app.delete("/api/history/<int:i>")
def delete(i):
    db.delete_assessment(i)
    return jsonify(ok=True)

@app.get("/api/history.csv")
def export():
    rows = db.list_assessments()
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=assessments.csv"})

@app.get("/api/model")
def model_info():
    with db.conn() as c:
        runs = [dict(r) for r in c.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 8")]
    return jsonify(metrics=METRICS, runs=runs)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", port=port, debug=False)
