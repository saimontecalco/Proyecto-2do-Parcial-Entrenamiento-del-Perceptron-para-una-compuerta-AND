"""perceptron.py
Lógica pura del Perceptrón Monocapa (SLP) para la compuerta AND.
No depende de Streamlit ni de ninguna librería de PDF.
"""
from dataclasses import dataclass, asdict
from typing import List, Tuple

# Tabla de verdad de la compuerta AND: ((x1, x2), t)
AND_DATA: List[Tuple[Tuple[int, int], int]] = [
    ((0, 0), 0),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1),
]

_PREC = 10  # redondeo para evitar ruido de punto flotante (0.1+0.2 = 0.30000000000000004)


def _r(v: float) -> float:
    return round(v + 0.0, _PREC)  # "+ 0.0" evita -0.0


def step(z: float) -> int:
    """Función de activación escalón: 1 si z >= 0, 0 en otro caso."""
    return 1 if z >= 0 else 0


@dataclass
class StepRecord:
    """Registro de una muestra evaluada dentro de una época."""
    epoch: int
    sample: int
    x1: float
    x2: float
    t: int
    w1_prev: float
    w2_prev: float
    b_prev: float
    z: float
    y: int
    e: int
    w1_new: float
    w2_new: float
    b_new: float

    @property
    def updated(self) -> bool:
        return self.e != 0


@dataclass
class EpochRecord:
    """Resumen al final de una época."""
    epoch: int
    w1: float
    w2: float
    b: float
    errors: int


class Perceptron:
    def __init__(self, w1: float, w2: float, b: float, eta: float):
        self.w1_init, self.w2_init, self.b_init = float(w1), float(w2), float(b)
        self.w1, self.w2, self.b = float(w1), float(w2), float(b)
        self.eta = float(eta)
        self.steps: List[StepRecord] = []
        self.epochs: List[EpochRecord] = []
        self.converged = False

    # ---------- inferencia ----------
    def weighted_sum(self, x1: float, x2: float) -> float:
        return _r(x1 * self.w1 + x2 * self.w2 + self.b)

    def predict(self, x1: float, x2: float) -> Tuple[float, int]:
        """Devuelve (z, y) con los pesos actuales."""
        z = self.weighted_sum(x1, x2)
        return z, step(z)

    # ---------- entrenamiento ----------
    def train(self, max_epochs: int, data=AND_DATA) -> "Perceptron":
        """Regla delta: w_new = w_prev + eta * e * x ; b_new = b_prev + eta * e.
        Se detiene cuando una época completa no tiene errores o al llegar a max_epochs."""
        self.steps, self.epochs, self.converged = [], [], False
        for epoch in range(1, int(max_epochs) + 1):
            errors = 0
            for i, ((x1, x2), t) in enumerate(data, start=1):
                w1p, w2p, bp = self.w1, self.w2, self.b
                z = self.weighted_sum(x1, x2)
                y = step(z)
                e = t - y
                if e != 0:
                    errors += 1
                self.w1 = _r(w1p + self.eta * e * x1)
                self.w2 = _r(w2p + self.eta * e * x2)
                self.b = _r(bp + self.eta * e)
                self.steps.append(StepRecord(epoch, i, x1, x2, t, w1p, w2p, bp,
                                             z, y, e, self.w1, self.w2, self.b))
            self.epochs.append(EpochRecord(epoch, self.w1, self.w2, self.b, errors))
            if errors == 0:
                self.converged = True
                break
        return self

    # ---------- utilidades ----------
    def steps_of_epoch(self, epoch: int) -> List[StepRecord]:
        return [s for s in self.steps if s.epoch == epoch]

    def epochs_as_dicts(self):
        return [asdict(e) for e in self.epochs]
