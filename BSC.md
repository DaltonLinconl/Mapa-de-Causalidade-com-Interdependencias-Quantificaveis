# 📊 BSC Simulado — Empresa Fictícia de Tecnologia

## 🔷 Estrutura Geral

Este Balanced Scorecard foi estruturado para modelagem causal quantitativa.
Contém 8 variáveis distribuídas nas 4 perspectivas clássicas.

---

# 🧭 1️⃣ Perspectiva de Aprendizado & Crescimento

Objetivo estratégico:
Desenvolver capacidade organizacional por meio de pessoas e tecnologia.

| Código | Variável                   | Descrição                                        |
| ------ | -------------------------- | ------------------------------------------------ |
| A1     | Horas de Treinamento       | Média de horas de treinamento por colaborador    |
| A2     | Investimento em Tecnologia | Investimento anual em infraestrutura tecnológica |

---

# ⚙️ 2️⃣ Perspectiva de Processos Internos

Objetivo estratégico:
Melhorar eficiência operacional e qualidade.

| Código | Variável               | Descrição                              |
| ------ | ---------------------- | -------------------------------------- |
| P1     | Eficiência Operacional | Índice de produtividade operacional    |
| P2     | Taxa de Retrabalho     | Percentual de retrabalho nos processos |

---

# 👥 3️⃣ Perspectiva de Clientes

Objetivo estratégico:
Aumentar valor percebido e fidelização.

| Código | Variável              | Descrição                                    |
| ------ | --------------------- | -------------------------------------------- |
| C1     | Satisfação do Cliente | Índice de satisfação (ex: NPS padronizado)   |
| C2     | Taxa de Retenção      | Percentual de clientes que permanecem ativos |

---

# 💰 4️⃣ Perspectiva Financeira

Objetivo estratégico:
Maximizar desempenho financeiro sustentável.

| Código | Variável           | Descrição                          |
| ------ | ------------------ | ---------------------------------- |
| F1     | Receita            | Receita total no período           |
| F2     | Margem Operacional | Percentual de margem sobre receita |

---

# 🔁 Estrutura Causal Definida

Relações direcionadas entre variáveis:

A1 → P1  
A2 → P1  
P1 → P2  
P1 → C1  
P2 → C1  
C1 → C2  
C2 → F1  
P1 → F1  
F1 → F2

---

# 🧠 Interpretação Estratégica

Aprendizado & Crescimento  
↓  
Processos Internos  
↓  
Clientes  
↓  
Financeiro

Interpretação:

- Mais treinamento e investimento tecnológico aumentam eficiência.
- Maior eficiência reduz retrabalho.
- Melhor desempenho operacional aumenta satisfação do cliente.
- Clientes satisfeitos permanecem mais tempo.
- Retenção e eficiência impactam receita.
- Receita impacta margem operacional.

---

# 📦 Estrutura Final do Dataset

Variáveis do modelo:

A1, A2, P1, P2, C1, C2, F1, F2

Cada observação representa um período ou unidade organizacional.

---

# 🎯 Finalidade do Modelo

Este BSC será utilizado para:

- Modelagem via Structural Equation Modeling (SEM)
- Estimativa de pesos causais
- Validação estatística das relações estratégicas
- Construção de grafo direcionado com interdependências quantificáveis
