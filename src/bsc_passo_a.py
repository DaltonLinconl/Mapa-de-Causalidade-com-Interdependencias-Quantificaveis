"""
BSC — Mapa de Causalidade com Interdependências Quantificáveis
═══════════════════════════════════════════════════════════════
PASSO A: Simulação do Dataset

Objetivo: gerar um dataset de N observações que respeita a estrutura
causal do BSC. As correlações entre variáveis emergem das relações
causais — não são impostas artificialmente.

Cada observação representa um período ou unidade organizacional.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Reprodutibilidade ────────────────────────────────────────────────────────
# Fixar a semente garante que você obtém os mesmos dados toda vez.
# Troque o valor para gerar amostras diferentes.
np.random.seed(42)

N = 200  # número de observações

print("=" * 60)
print("PASSO A — Simulação do Dataset BSC")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# BLOCO 1 — Perspectiva de Aprendizado & Crescimento
# Variáveis raiz: não têm predecessores no grafo.
# São geradas como puro ruído gaussiano com parâmetros realistas.
# ════════════════════════════════════════════════════════════════

A1 = np.random.normal(loc=32, scale=6, size=N)
#    └─ Horas de Treinamento: média 32h, desvio 6h

A2 = np.random.normal(loc=850, scale=150, size=N)
#    └─ Investimento em Tecnologia: média R$850k, desvio R$150k

print("\n[A] Aprendizado & Crescimento")
print(f"    A1 (Horas Treinamento)   → média: {A1.mean():.1f}h  | dp: {A1.std():.1f}h")
print(
    f"    A2 (Invest. Tecnologia)  → média: R${A2.mean():.0f}k | dp: R${A2.std():.0f}k"
)

# ════════════════════════════════════════════════════════════════
# PADRONIZAÇÃO das variáveis de entrada
#
# Por quê? A1 está em horas (escala ~30) e A2 em R$k (escala ~850).
# Se usarmos as escalas brutas, os β não serão comparáveis.
# Padronizar (z-score) coloca tudo em desvios padrão.
#
# z = (x - média) / desvio_padrão
# Resultado: média 0, desvio padrão 1
# ════════════════════════════════════════════════════════════════


def zscale(x):
    return (x - x.mean()) / x.std()


A1_s = zscale(A1)
A2_s = zscale(A2)

# ════════════════════════════════════════════════════════════════
# BLOCO 2 — Perspectiva de Processos Internos
# ════════════════════════════════════════════════════════════════

# P1 (Eficiência Operacional) ← A1, A2
# Interpretação dos β:
#   +0.38: mais treinamento → mais eficiência
#   +0.31: mais tecnologia  → mais eficiência
# O ruído (scale=0.04) representa fatores não modelados
P1 = (
    0.74  # intercepto (nível base de eficiência)
    + 0.38 * A1_s  # efeito do treinamento
    + 0.31 * A2_s  # efeito da tecnologia
    + np.random.normal(0, 0.04, N)
)  # ruído
P1 = np.clip(P1, 0.40, 0.99)  # eficiência é um índice [0,1]

# P2 (Taxa de Retrabalho) ← P1
# β NEGATIVO: mais eficiência → MENOS retrabalho (relação inversa)
P2 = (
    0.35
    - 0.55 * zscale(P1)  # β negativo = relação inversa
    + np.random.normal(0, 0.02, N)
)
P2 = np.clip(P2, 0.01, 0.50)

print("\n[P] Processos Internos")
print(f"    P1 (Eficiência Operacional) → média: {P1.mean():.3f} | dp: {P1.std():.3f}")
print(f"    P2 (Taxa de Retrabalho)     → média: {P2.mean():.3f} | dp: {P2.std():.3f}")

# ════════════════════════════════════════════════════════════════
# BLOCO 3 — Perspectiva de Clientes
# ════════════════════════════════════════════════════════════════

# C1 (Satisfação do Cliente) ← P1, P2
# P1 tem β positivo (eficiência melhora satisfação)
# P2 tem β negativo (retrabalho prejudica satisfação)
C1 = 0.30 + 0.44 * zscale(P1) - 0.28 * zscale(P2) + np.random.normal(0, 0.03, N)
C1 = np.clip(C1, 0.30, 0.99)

# C2 (Taxa de Retenção) ← C1
# Clientes satisfeitos ficam → relação positiva forte
C2 = 0.20 + 0.62 * zscale(C1) + np.random.normal(0, 0.03, N)
C2 = np.clip(C2, 0.30, 0.99)

print("\n[C] Clientes")
print(f"    C1 (Satisfação do Cliente) → média: {C1.mean():.3f} | dp: {C1.std():.3f}")
print(f"    C2 (Taxa de Retenção)      → média: {C2.mean():.3f} | dp: {C2.std():.3f}")

# ════════════════════════════════════════════════════════════════
# BLOCO 4 — Perspectiva Financeira
# ════════════════════════════════════════════════════════════════

# F1 (Receita) ← C2, P1
# Retenção de clientes e eficiência impactam receita
# Escala grande: R$ mil
F1 = 1500 + 1800 * zscale(C2) + 900 * zscale(P1) + np.random.normal(0, 200, N)
F1 = np.clip(F1, 500, 8000)

# F2 (Margem Operacional) ← F1
# Mais receita → mais margem (economias de escala)
F2 = 0.05 + 0.71 * zscale(F1) + np.random.normal(0, 0.03, N)
F2 = np.clip(F2, 0.01, 0.60)

print("\n[F] Financeiro")
print(
    f"    F1 (Receita)              → média: R${F1.mean():.0f}k | dp: R${F1.std():.0f}k"
)
print(f"    F2 (Margem Operacional)   → média: {F2.mean():.3f} | dp: {F2.std():.3f}")

# ════════════════════════════════════════════════════════════════
# BLOCO 5 — Montar e inspecionar o DataFrame
# ════════════════════════════════════════════════════════════════

df = pd.DataFrame(
    {
        "A1": A1,
        "A2": A2,  # Aprendizado
        "P1": P1,
        "P2": P2,  # Processos
        "C1": C1,
        "C2": C2,  # Clientes
        "F1": F1,
        "F2": F2,  # Financeiro
    }
)

print(f"\n{'=' * 60}")
print(f"Dataset gerado: {df.shape[0]} observações × {df.shape[1]} variáveis")
print(f"{'=' * 60}")
print("\nEstatísticas descritivas:")
print(df.describe().round(3))

# ════════════════════════════════════════════════════════════════
# BLOCO 6 — Verificação: a matriz de correlação
#
# Se a simulação está correta, devemos observar:
#   • A1 e A2 correlacionados positivamente com P1
#   • P2 correlacionado NEGATIVAMENTE com P1 (relação inversa)
#   • C1 e C2 correlacionados positivamente com P1
#   • F1 e F2 correlacionados positivamente com C2 e P1
# ════════════════════════════════════════════════════════════════

print("\nMatriz de Correlação (valores esperados vs. observados):")
corr = df.corr().round(3)
print(corr)

# ════════════════════════════════════════════════════════════════
# BLOCO 7 — Visualização: 3 painéis diagnósticos
# ════════════════════════════════════════════════════════════════

DARK = "#070910"
DARK2 = "#0d1017"
DARK3 = "#111620"
BORDER = "#1e2535"
TEXT = "#e2e8f0"
MUTED = "#4a5568"

COLORS = {
    "A": "#3b82f6",  # Aprendizado — azul
    "P": "#8b5cf6",  # Processos   — roxo
    "C": "#10b981",  # Clientes    — verde
    "F": "#f59e0b",  # Financeiro  — âmbar
}


def node_color(col):
    return COLORS[col[0]]


fig = plt.figure(figsize=(18, 14), facecolor=DARK)
gs = gridspec.GridSpec(
    3,
    2,
    figure=fig,
    hspace=0.45,
    wspace=0.35,
    left=0.07,
    right=0.96,
    top=0.93,
    bottom=0.06,
)

# ── Painel 1: Correlação ─────────────────────────────────────────
ax_corr = fig.add_subplot(gs[0, :])  # ocupa linha inteira
ax_corr.set_facecolor(DARK2)

mask = np.zeros_like(corr, dtype=bool)  # sem máscara = mostrar tudo
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(
    corr,
    ax=ax_corr,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor=BORDER,
    annot_kws={"size": 9, "color": TEXT},
    cbar_kws={"shrink": 0.8},
)

ax_corr.set_title(
    "Matriz de Correlação — verificação da estrutura causal",
    color=TEXT,
    fontsize=12,
    pad=12,
)
ax_corr.tick_params(colors=MUTED, labelsize=10)
ax_corr.collections[0].colorbar.ax.tick_params(colors=MUTED)

# Colorir labels dos eixos por perspectiva
for label in ax_corr.get_xticklabels() + ax_corr.get_yticklabels():
    label.set_color(node_color(label.get_text()))
    label.set_fontweight("bold")

# ── Painel 2: Distribuições ──────────────────────────────────────
cols_to_plot = ["A1", "P1", "P2", "C1", "F1", "F2"]
axes_dist = [fig.add_subplot(gs[1, i // 3 * 1 + i % 3 // 3]) for i in range(2)]

# vamos fazer subgrid manual para 6 histogramas em 2x3
inner_gs1 = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs[1, :], hspace=0.55, wspace=0.35
)

for idx, col in enumerate(cols_to_plot):
    ax = fig.add_subplot(inner_gs1[idx // 3, idx % 3])
    ax.set_facecolor(DARK3)
    color = node_color(col)
    ax.hist(df[col], bins=30, color=color, alpha=0.8, edgecolor="none")
    ax.set_title(col, color=color, fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[:].set_color(BORDER)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    # linha da média
    ax.axvline(
        df[col].mean(),
        color="white",
        linewidth=1,
        linestyle="--",
        alpha=0.5,
        label=f"μ={df[col].mean():.2f}",
    )
    ax.legend(
        fontsize=7,
        facecolor=DARK2,
        edgecolor=BORDER,
        labelcolor=TEXT,
        loc="upper right",
    )

# ── Painel 3: Relações chave ─────────────────────────────────────
scatter_pairs = [
    ("A1", "P1", "A1 → P1  (β=+0.38 esperado)"),
    ("P1", "P2", "P1 → P2  (β=−0.55 esperado)"),
    ("C1", "C2", "C1 → C2  (β=+0.62 esperado)"),
]

inner_gs2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs[2, :], hspace=0.4, wspace=0.35
)

for idx, (x_col, y_col, title) in enumerate(scatter_pairs):
    ax = fig.add_subplot(inner_gs2[idx])
    ax.set_facecolor(DARK3)
    cx = node_color(x_col)

    ax.scatter(df[x_col], df[y_col], alpha=0.35, s=18, color=cx, edgecolors="none")

    # linha de tendência
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    ax.plot(x_line, p(x_line), color="white", linewidth=1.5, alpha=0.8)

    # correlação observada
    r = df[[x_col, y_col]].corr().iloc[0, 1]
    ax.text(
        0.05,
        0.93,
        f"r = {r:.3f}",
        transform=ax.transAxes,
        color=TEXT,
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(facecolor=DARK2, edgecolor=BORDER, boxstyle="round,pad=0.3"),
    )

    ax.set_xlabel(x_col, color=cx, fontsize=9)
    ax.set_ylabel(y_col, color=node_color(y_col), fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=9, pad=8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[:].set_color(BORDER)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

fig.suptitle(
    "PASSO A — Diagnóstico do Dataset Simulado",
    color=TEXT,
    fontsize=15,
    fontweight="bold",
    y=0.97,
)

output_path = "src/bsc_passo_a_diagnostico.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"\nGráfico salvo em: {output_path}")

# Salvar o dataset para os próximos passos
csv_path = "src/bsc_dataset.csv"
df.to_csv(csv_path, index=False)
print(f"Dataset salvo em: {csv_path}")
