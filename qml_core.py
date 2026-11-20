"""Variational quantum classifier (VQC) with data re-uploading, built on PennyLane."""
import numpy as np
import pennylane as qml

FEATURES = ["Glucose", "BMI", "Age", "BloodPressure", "DiabetesPedigreeFunction", "Pregnancies"]
N_QUBITS = len(FEATURES)
MISSING_AS_ZERO = ["Glucose", "BloodPressure", "BMI"]

dev = qml.device("default.qubit", wires=N_QUBITS)

def make_circuit(n_layers):
    @qml.qnode(dev, interface="autograd")
    def circuit(x, w, s):
        for l in range(n_layers):
            qml.AngleEmbedding(x * s[l], wires=range(N_QUBITS), rotation="Y")   # encode patient features
            qml.StronglyEntanglingLayers(w[l:l + 1], wires=range(N_QUBITS))  # trainable entangling block
        return qml.expval(qml.PauliZ(0))
    return circuit

class Preprocessor:
    """Median-imputes physiologically impossible zeros, then maps each feature to [0, pi]."""
    def __init__(self, med=None, lo=None, hi=None):
        self.med, self.lo, self.hi = med, lo, hi

    def fit(self, df):
        d = df[FEATURES].astype(float).copy()
        self.med = {c: float(d[c][d[c] > 0].median()) for c in MISSING_AS_ZERO}
        d = self._impute(d)
        self.lo = {c: float(d[c].quantile(0.01)) for c in FEATURES}
        self.hi = {c: float(d[c].quantile(0.99)) for c in FEATURES}
        return self

    def _impute(self, d):
        d = d.copy()
        for c in MISSING_AS_ZERO:
            d.loc[d[c] <= 0, c] = self.med[c]
        return d

    def transform(self, df):
        d = self._impute(df[FEATURES].astype(float))
        out = np.stack([(d[c].values - self.lo[c]) / (self.hi[c] - self.lo[c]) for c in FEATURES], axis=1)
        return np.clip(out, 0, 1) * np.pi

    def to_dict(self):
        return {"med": self.med, "lo": self.lo, "hi": self.hi}

class QuantumClassifier:
    def __init__(self, w, a, b, n_layers, s=None):
        self.s = np.array(s if s is not None else np.ones((n_layers, N_QUBITS)))
        self.w, self.a, self.b, self.n_layers = np.array(w), float(a), float(b), n_layers
        self.circuit = make_circuit(n_layers)

    def proba(self, X):
        z = np.array(self.circuit(np.array(X), self.w, self.s))
        return 1 / (1 + np.exp(-(-self.a * z + self.b)))
