"""Train the quantum classifier + classical baselines on data stored in SQLite."""
import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
import db
from qml_core import Preprocessor, QuantumClassifier, make_circuit, FEATURES, N_QUBITS

OUT = os.path.join(db.ROOT, "model")

def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    return dict(accuracy=accuracy_score(y, yh), roc_auc=roc_auc_score(y, p), f1=f1_score(y, yh),
                recall=recall_score(y, yh), precision=precision_score(y, yh),
                confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))

def train(epochs=40, layers=3, batch=48, lr=0.08, seed=7, log=print):
    db.init_db()
    df = db.load_dataset()
    tr, te = train_test_split(df, test_size=0.2, stratify=df.Outcome, random_state=42)
    pre = Preprocessor().fit(tr)
    Xtr, Xte = pre.transform(tr), pre.transform(te)
    ytr, yte = tr.Outcome.values, te.Outcome.values

    rng = np.random.default_rng(seed)
    w = pnp.array(rng.normal(0, 0.3, (layers, N_QUBITS, 3)), requires_grad=True)
    s = pnp.array(0.45 * np.ones((layers, N_QUBITS)) + rng.normal(0, 0.03, (layers, N_QUBITS)), requires_grad=True)
    a = pnp.array(3.0, requires_grad=True)
    b = pnp.array(0.0, requires_grad=True)
    circuit = make_circuit(layers)
    pw = float((ytr == 0).sum() / (ytr == 1).sum())  # up-weight diabetic cases: screening favours recall

    def loss(w, s, a, b, X, y):
        p = 1 / (1 + pnp.exp(-(-a * circuit(X, w, s) + b)))
        return -pnp.mean(pw * y * pnp.log(p + 1e-7) + (1 - y) * pnp.log(1 - p + 1e-7)) + 0.05 * pnp.mean(s ** 2)

    opt = qml.AdamOptimizer(lr)
    history, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(Xtr)); losses = []
        for st in range(0, len(idx), batch):
            j = idx[st:st + batch]
            Xb, yb = pnp.array(Xtr[j], requires_grad=False), pnp.array(ytr[j], requires_grad=False)
            (w, s, a, b), l = opt.step_and_cost(lambda w, s, a, b: loss(w, s, a, b, Xb, yb), w, s, a, b)
            losses.append(float(l))
        history.append(float(np.mean(losses)))
        if ep % 5 == 0 or ep == 1:
            log(f"epoch {ep:3d}/{epochs}  loss {history[-1]:.4f}  ({time.time()-t0:.0f}s)")

    qc = QuantumClassifier(w.numpy(), float(a), float(b), layers, s.numpy())
    results = {"Quantum VQC": metrics(yte, qc.proba(Xte))}
    results["Quantum VQC"]["loss_history"] = history

    Xtr_c, Xte_c = tr[FEATURES].values, te[FEATURES].values
    baselines = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=0)),
        "Random Forest": RandomForestClassifier(200, class_weight="balanced", max_depth=6, random_state=0),
    }
    for name, m in baselines.items():
        m.fit(Xtr_c, ytr)
        results[name] = metrics(yte, m.predict_proba(Xte_c)[:, 1])

    os.makedirs(OUT, exist_ok=True)
    json.dump({"w": w.numpy().tolist(), "s": s.numpy().tolist(), "a": float(a), "b": float(b), "layers": layers,
               "pre": pre.to_dict()}, open(os.path.join(OUT, "qml_weights.json"), "w"))
    summary = {"results": results, "n_train": len(tr), "n_test": len(te), "qubits": N_QUBITS, "layers": layers,
               "params": int(w.size + s.size + 2), "epochs": epochs, "features": FEATURES,
               "base_rate": float(df.Outcome.mean()), "dataset_size": len(df)}
    json.dump(summary, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    for k, v in results.items():
        db.save_run(k, v, f"{epochs} epochs" if k.startswith("Quantum") else "")
        log(f"{k:22s} acc {v['accuracy']:.3f}  auc {v['roc_auc']:.3f}  recall {v['recall']:.3f}  f1 {v['f1']:.3f}")
    return summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    train(a.epochs, a.layers)
