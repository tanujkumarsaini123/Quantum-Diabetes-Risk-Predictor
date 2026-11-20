# Qubit Glucose: early diabetes risk prediction with quantum machine learning

A complete, runnable project: a 6-qubit **variational quantum classifier** (PennyLane) trained on the Pima Indians Diabetes dataset, served through a Flask web app, with all data stored in **SQLite**.

## Run it (VS Code)
1. Open this folder in VS Code (File > Open Folder).
2. Open the terminal (Ctrl+`) and run:
   - Windows: `run.bat`
   - macOS/Linux: `./run.sh`
   
   Or manually: `python -m venv .venv`, activate it, `pip install -r requirements.txt`, `python app.py`.
3. Your browser opens at http://127.0.0.1:5000 (Python 3.10+ required).

A trained model is already included, so the app starts instantly. To retrain (about 2 minutes): `python train.py --epochs 100 --layers 3`.

## How it works
| Part | File | What it does |
|---|---|---|
| Quantum model | `qml_core.py` | 6 qubits (Glucose, BMI, Age, Blood pressure, Family history, Pregnancies). Each layer: RY angle encoding with trainable input scale, then StronglyEntanglingLayers. Output is the Pauli-Z expectation on qubit 1, mapped to a probability. |
| Training | `train.py` | Loads data **from SQLite**, imputes impossible zeros, scales to [0, pi], trains with Adam and class-weighted cross-entropy plus an L2 penalty on input scales (keeps risk curves smooth). Compares against Logistic Regression, SVM and Random Forest. |
| Database | `db.py` | SQLite tables: `dataset` (768 records), `training_runs` (model registry), `assessments` (saved patient results). File: `diabetes.db`, created automatically. |
| API + UI | `app.py`, `templates/`, `static/` | Assess risk, patient history (with delete and CSV export), and a Model lab showing the circuit, metrics, loss curve and confusion matrix. |

## Results (154 held-out patients)
| Model | ROC AUC | Recall | Accuracy | F1 |
|---|---|---|---|---|
| Quantum VQC | 0.81 | 0.80 | 0.72 | 0.67 |
| Logistic Regression | 0.82 | 0.74 | 0.74 | 0.67 |
| SVM (RBF) | 0.80 | 0.61 | 0.75 | 0.63 |
| Random Forest | 0.82 | 0.74 | 0.75 | 0.68 |

The quantum model is competitive with classical baselines, not better. On a small tabular dataset that is the honest expectation; it runs on a classical simulator (`default.qubit`), not quantum hardware. To target real devices, swap the device in `qml_core.py` for a PennyLane plugin (IBM, IonQ, etc.).

## Explainability
For each prediction the app re-runs the circuit with one feature at a time replaced by the population median; the change in risk is shown per qubit.

## Limitations
- Dataset: 768 women of Pima heritage aged 21+. Results do not generalise to everyone.
- The risk curve for Age is not perfectly monotonic (one small dip in testing); other features behave smoothly.
- Educational prototype, not a medical device or a diagnosis.

## Ideas to extend
Add a hardware backend, cross-validation, SHAP-style explanations, user logins, or PDF reports.
