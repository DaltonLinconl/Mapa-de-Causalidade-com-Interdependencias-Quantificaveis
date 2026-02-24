import numpy as np
import pandas as pd

np.random.seed(42)
N = 200  # observações (ex: 200 períodos ou unidades organizacionais)

# ── Causas raiz (sem predecessores) ──────────────────────────────────────────
A1 = np.random.normal(loc=32, scale=6, size=N)  # horas de treinamento
A2 = np.random.normal(loc=850, scale=150, size=N)  # investimento em tecnologia (R$ k)

# ── Processos Internos ────────────────────────────────────────────────────────
# P1 depende de A1 e A2
# Padronizamos A1 e A2 para que os betas sejam comparáveis entre si
A1_s = (A1 - A1.mean()) / A1.std()
A2_s = (A2 - A2.mean()) / A2.std()

P1 = 0.74 + 0.08 * A1_s + 0.06 * A2_s + np.random.normal(0, 0.04, N)
P1 = np.clip(P1, 0.40, 0.99)  # índice entre 0 e 1

# P2 depende de P1 (retrabalho diminui quando eficiência aumenta)
P2 = 0.35 - 0.28 * P1 + np.random.normal(0, 0.02, N)
P2 = np.clip(P2, 0.01, 0.50)

# ── Clientes ──────────────────────────────────────────────────────────────────
C1 = 0.30 + 0.44 * P1 - 0.18 * P2 + np.random.normal(0, 0.03, N)
C1 = np.clip(C1, 0.30, 0.99)

C2 = 0.20 + 0.62 * C1 + np.random.normal(0, 0.03, N)
C2 = np.clip(C2, 0.30, 0.99)

# ── Financeiro ────────────────────────────────────────────────────────────────
F1 = 1500 + 1800 * C2 + 900 * P1 + np.random.normal(0, 200, N)  # Receita (R$ k)
F1 = np.clip(F1, 500, 8000)

F2 = 0.05 + 0.71 * (F1 / F1.max()) + np.random.normal(0, 0.03, N)  # Margem
F2 = np.clip(F2, 0.01, 0.60)

# ── Montar DataFrame ──────────────────────────────────────────────────────────
df = pd.DataFrame(
    {
        "A1": A1,
        "A2": A2,
        "P1": P1,
        "P2": P2,
        "C1": C1,
        "C2": C2,
        "F1": F1,
        "F2": F2,
    }
)

print(df.describe().round(3))
