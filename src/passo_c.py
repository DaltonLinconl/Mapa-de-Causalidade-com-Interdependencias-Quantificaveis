import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


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


# ── Definir o grafo ───────────────────────────────────────────────────────────
G = nx.DiGraph()  # DiGraph = Directed Graph

# Adicionar nós com atributo de perspectiva
nodes = {
    "A1": "Aprendizado",
    "A2": "Aprendizado",
    "P1": "Processos",
    "P2": "Processos",
    "C1": "Clientes",
    "C2": "Clientes",
    "F1": "Financeiro",
    "F2": "Financeiro",
}
for node, persp in nodes.items():
    G.add_node(node, perspectiva=persp)

# Adicionar arestas (sem peso por enquanto)
edges = [
    ("A1", "P1"),
    ("A2", "P1"),
    ("P1", "P2"),
    ("P1", "C1"),
    ("P2", "C1"),
    ("C1", "C2"),
    ("C2", "F1"),
    ("P1", "F1"),
    ("F1", "F2"),
]
G.add_edges_from(edges)

# ── Layout por perspectiva (posições fixas) ───────────────────────────────────
pos = {
    "A1": (0, 2),
    "A2": (0, 0),  # coluna 0 — Aprendizado
    "P1": (2, 1),
    "P2": (3, 0),  # coluna 2-3 — Processos
    "C1": (4, 2),
    "C2": (5, 1),  # coluna 4-5 — Clientes
    "F1": (7, 2),
    "F2": (7, 0),  # coluna 7 — Financeiro
}

# ── Cores por perspectiva ─────────────────────────────────────────────────────
color_map = {
    "Aprendizado": "#3b82f6",
    "Processos": "#8b5cf6",
    "Clientes": "#10b981",
    "Financeiro": "#f59e0b",
}
node_colors = [color_map[nodes[n]] for n in G.nodes()]

# ── Visualização ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#070910")
ax.set_facecolor("#0d1017")

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, alpha=0.9, ax=ax)

nx.draw_networkx_labels(
    G, pos, font_color="white", font_size=12, font_weight="bold", ax=ax
)

nx.draw_networkx_edges(
    G,
    pos,
    edge_color="#4a6fa5",
    arrows=True,
    arrowsize=20,
    width=1.5,
    ax=ax,
    connectionstyle="arc3,rad=0.05",
)

# Legenda
legend_handles = [mpatches.Patch(color=c, label=p) for p, c in color_map.items()]
ax.legend(
    handles=legend_handles,
    loc="upper left",
    facecolor="#111620",
    edgecolor="#1e2535",
    labelcolor="white",
    fontsize=9,
)

ax.set_title("BSC — Grafo Causal Dirigido", color="white", fontsize=14, pad=16)
ax.axis("off")
plt.tight_layout()
plt.show()


# ── Padronizar o dataset ──────────────────────────────────────────────────────
scaler = StandardScaler()
df_s = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# ── Estrutura causal: quem são os predecessores de cada nó? ──────────────────
predecessors = {
    "P1": ["A1", "A2"],
    "P2": ["P1"],
    "C1": ["P1", "P2"],
    "C2": ["C1"],
    "F1": ["C2", "P1"],
    "F2": ["F1"],
}

# ── Estimar regressão para cada nó ───────────────────────────────────────────
edge_weights = {}

for target, sources in predecessors.items():
    X = df_s[sources].values
    y = df_s[target].values

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    print(f"\n{target} ~ {' + '.join(sources)}")
    print(f"  R² = {r2:.3f}")

    for source, coef in zip(sources, model.coef_):
        key = f"{source}→{target}"
        edge_weights[key] = round(coef, 3)
        print(f"  β({source}) = {coef:.3f}")

print("\n\nPesos estimados por aresta:")
for k, v in edge_weights.items():
    print(f"  {k}: {v:+.3f}")


# ── Adicionar pesos ao grafo ──────────────────────────────────────────────────
for edge_key, weight in edge_weights.items():
    src, tgt = edge_key.split("→")
    G[src][tgt]["weight"] = weight

# ── Preparar visualização com pesos ──────────────────────────────────────────
weights = [abs(G[u][v].get("weight", 0.1)) for u, v in G.edges()]
edge_colors = [
    "#ef4444" if G[u][v].get("weight", 0) < 0 else "#4a9eff" for u, v in G.edges()
]
edge_widths = [w * 6 + 0.5 for w in weights]  # espessura proporcional ao peso

edge_labels = {
    (src, tgt): f"{G[src][tgt]['weight']:+.2f}"
    for src, tgt in G.edges()
    if "weight" in G[src][tgt]
}

# ── Plot final ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("#070910")
ax.set_facecolor("#0d1017")

# Fundo por perspectiva (zonas visuais)
zones = {
    "Aprendizado & Crescimento": ((-0.8, -0.7), 1.3, 3.4),
    "Processos Internos": ((1.2, -0.7), 2.5, 3.4),
    "Clientes": ((3.7, -0.7), 2.0, 3.4),
    "Financeiro": ((6.2, -0.7), 1.5, 3.4),
}
zone_colors = ["#3b82f611", "#8b5cf611", "#10b98111", "#f59e0b11"]
for (label, ((x, y), w, h)), color in zip(zones.items(), zone_colors):
    rect = plt.Rectangle((x, y), w, h, linewidth=0, facecolor=color, zorder=0)
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h + 0.05,
        label,
        ha="center",
        color="#4a5568",
        fontsize=8,
        style="italic",
    )

# Nós
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=2200,
    alpha=0.95,
    ax=ax,
    linewidths=2,
    edgecolors="#ffffff22",
)

nx.draw_networkx_labels(
    G, pos, font_color="white", font_size=13, font_weight="bold", ax=ax
)

# Arestas com espessura e cor por sinal
nx.draw_networkx_edges(
    G,
    pos,
    edge_color=edge_colors,
    width=edge_widths,
    arrows=True,
    arrowsize=18,
    ax=ax,
    alpha=0.85,
    connectionstyle="arc3,rad=0.05",
)

# Labels das arestas (pesos)
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=8,
    font_color="#cbd5e1",
    bbox=dict(
        boxstyle="round,pad=0.2", facecolor="#0d1017", edgecolor="none", alpha=0.8
    ),
    ax=ax,
)

# Legenda
legend_handles = [mpatches.Patch(color=c, label=p) for p, c in color_map.items()]
legend_handles += [
    mpatches.Patch(color="#4a9eff", label="β positivo"),
    mpatches.Patch(color="#ef4444", label="β negativo"),
]
ax.legend(
    handles=legend_handles,
    loc="lower right",
    facecolor="#111620",
    edgecolor="#1e2535",
    labelcolor="white",
    fontsize=9,
    ncol=2,
)

ax.set_title(
    "BSC — Grafo Causal com Pesos Estimados (OLS Padronizado)",
    color="white",
    fontsize=14,
    pad=20,
)
ax.set_xlim(-1, 8.5)
ax.set_ylim(-1, 3.5)
ax.axis("off")
plt.tight_layout()
plt.savefig("bsc_causal_graph.png", dpi=150, bbox_inches="tight", facecolor="#070910")
plt.show()
