Claro! Aqui está o mesmo conteúdo **com formatação ajustada** — mantendo **todas as palavras originais**, apenas melhorando espaçamento, hierarquia e legibilidade em Markdown:

---

# 🎮 MLP – Estilos de Jogo no LoL Worlds 2024

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-MLPClassifier-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-data%20analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-graphs-informational)](https://matplotlib.org/)
[![Kaggle Dataset](https://img.shields.io/badge/dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/anmatngu/2024-lol-championship-player-stats-and-swiss-stage)

---

Este projeto treina uma rede neural artificial (MLP) para descobrir e classificar estilos de jogo de jogadores profissionais do **League of Legends World Championship 2024**.

O modelo usa Inteligência Artificial para analisar estatísticas reais do Worlds 2024 e identificar padrões de estilo de jogo entre profissionais.
Além de classificar jogadores, a IA também calcula a **sinergia de cada time** (combinação de estilos + desempenho médio) para prever o **Top 4 técnico** e um **MVP IA**.

---

## 📚 Sumário rápido

- [Como foi feito](#-o-que-foi-feito-explicação-simples)
- [Estilos criados](#-os-estilos-de-jogo-criados)
- [Critérios de classificação](#-critérios-de-classificação-por-estilo)
- [Sinergia de time e Top 4 IA](#-sinergia-de-time-e-campeão-ia)
- [Como rodar o projeto](#-como-usar)
- [Resultados e gráficos](#-entendendo-os-resultados)
- [Créditos e licença](#-autoria-e-creditos)

---

## 🧠 O que foi feito (explicação simples)

1. Foram coletados dados reais de **81 jogadores** do campeonato
   (fonte: [Kaggle Dataset](https://www.kaggle.com/datasets/anmatngu/2024-lol-championship-player-stats-and-swiss-stage)).
2. As estatísticas foram tratadas (remoção de %, vírgulas, NaNs) e **padronizadas**.
3. Foram criados **rótulos de estilo de jogo** — “Agressivo”, “Carregador”, “Visionário” etc.
4. Foi treinada uma **MLP (Multi-Layer Perceptron)** do Scikit-learn.
5. Avaliação com **Hold-out** e **Validação cruzada (5-fold)**.
6. Foram gerados **gráficos e relatórios automáticos**.

---

## 🔍 O que é uma MLP?

A **MLP (Multi-Layer Perceptron)** é uma rede neural totalmente conectada que aprende padrões a partir de exemplos.
Neste projeto, ela recebe estatísticas como **KDA, DPM, GPM, KP%, visão, Solo Kills, GD@15** e aprende a associá-las a um **estilo de jogo primário**.

---

## 🧩 Os estilos de jogo criados

| Estilo             | Explicação simples                                       |
| :----------------- | :------------------------------------------------------- |
| 🗡️ **Agressivo**  | Parte pra cima, busca abates e pressiona o mapa.         |
| 💪 **Carregador**  | Principal fonte de dano e vitórias do time (carry).      |
| 🧱 **Consistente** | Joga de forma segura, erra pouco, mantém bom desempenho. |
| ⚔️ **Duelista**    | Forte em lutas 1x1, depende da mecânica individual.      |
| ⚖️ **Equilibrado** | Mistura ataque e defesa, joga de forma adaptável.        |
| 💥 **Volátil**     | Instável: pode jogar muito bem ou muito mal.             |
| 🩹 **Suporte**     | Ajuda o time com visão, cura, proteção e controle.       |
| 🔮 **Visionário**  | Foca em controle de mapa e visão estratégica.            |

Cada jogador pode ter **múltiplos estilos**. Para treinar, é escolhido um **Estilo Primário**.

---

## 🧩 Critérios de Classificação por Estilo

| Estilo             | Regra (simplificada)                             | Interpretação breve                 |
| :----------------- | :----------------------------------------------- | :---------------------------------- |
| 💪 **Carregador**  | DPM > p75 ∧ GPM > média ∧ KDA > média            | Dano alto, bom ouro e poucas mortes |
| 🗡️ **Agressivo**  | DPM > média ∧ (KP% > média ∨ Solo Kills > média) | Cria jogadas                        |
| 🔮 **Visionário**  | VSPM > média ∧ WPM > média ∧ DPM < média         | Foco em controle e visão            |
| 🩹 **Suporte**     | KP% > média ∧ WPM > média ∧ GPM < média          | Participativo e protetor            |
| 🧱 **Consistente** | KDA > p75 ∧ Avg Deaths < média                   | Estável e difícil de punir          |
| 💥 **Volátil**     | GD@15 < 0 ∧ Avg Deaths > média                   | Oscilante                           |
| ⚔️ **Duelista**    | Solo Kills > p75 ∧ DPM > média                   | Forte 1x1                           |
| ⚖️ **Equilibrado** | Nenhuma das regras acima                         | Meio-termo                          |

**Prioridade:**
Carregador > Agressivo > Visionário > Suporte > Consistente > Volátil > Duelista > Equilibrado.

---

## 🤝 Sinergia de Time e Campeão IA

**Synergy Score = 0,7 · StyleScore + 0,3 · PerfScore**

**Top 4 IA (sinergia estilo + performance)**

1. **Weibo Gaming** — 4,074
2. **T1** — 4,063
3. **Gen.G** — 3,617
4. **Team Liquid** — 3,487

**MVP IA:** `xiaohu` (Weibo Gaming) — Carregador, Agressivo, Consistente | DPM = 638.0

**Top 4 real:**
🥇 T1 | 🥈 BLG | 🥉–4º Weibo Gaming & Gen.G (sem disputa de 3º lugar)

A IA acertou **3 dos 4 times** do Top 4 real, apenas trocando a ordem.

---

## ⚙️ Como usar

```powershell
pip install -r requirements.txt
python .\src\mlp_estilo_lol_final.py
```

O script lê a base, calcula estilos, treina a MLP e gera relatórios em `outputs/`.

---

## 🗂️ Estrutura do Projeto

```
📁 src/
 ├── mlp_estilo_lol_final.py
 ├── player_statistics_cleaned_final.csv
📁 outputs/
 ├── confusion_matrix_estilo_*.png
 ├── estilos_bar_*.png
 ├── estilos_pie_*.png
 ├── estilos_multi_bar_*.png
 ├── estilos_multi_pie_*.png
 ├── relatorio_estilos_*.txt
 └── predicoes_completas_*.csv
```

---

## 📈 Entendendo os resultados

### ✅ Métricas

* **Acurácia (hold-out):** 0,6800
* **Precisão ponderada:** 0,7326
* **Recall ponderado:** 0,6800
* **F1 ponderado:** 0,6911
* **Validação cruzada (5 folds):** média 0,7412 ± 0,0586

---

### 🧾 Métricas explicadas

| Métrica      | Significado                            |
| :----------- | :------------------------------------- |
| **Acurácia** | Percentual total de acertos.           |
| **Precisão** | O quanto o estilo previsto está certo. |
| **Recall**   | Quantos reais a IA identifica.         |
| **F1-Score** | Equilíbrio entre precisão e recall.    |

---

### 💬 Exemplo de relatório

```
Acurácia: 0.80
Precisão média: 0.86
Estilo mais comum previsto: Agressivo
Time mais equilibrado: Weibo Gaming
MVP segundo a IA: xiaohu (Weibo Gaming)
Campeão real: T1 🏆
```

---

## 🖼️ Gráficos

| Arquivo                         | Mostra                  | Interpretação                    |
| :------------------------------ | :---------------------- | :------------------------------- |
| `confusion_matrix_estilo_*.png` | Matriz de confusão      | Acertos na diagonal              |
| `estilos_bar_*.png`             | Distribuição de estilos | Frequência de cada classe        |
| `estilos_pie_*.png`             | Proporção de estilos    | Percentual de cada estilo        |
| `estilos_multi_bar_*.png`       | Multiestilo (barras)    | Frequência de estilos combinados |
| `estilos_multi_pie_*.png`       | Multiestilo (pizza)     | Percentuais combinados           |
| `predicoes_completas_*.csv`     | Tabela de previsões     | Estilos por jogador              |
| `relatorio_estilos_*.txt`       | Relatório completo      | Métricas e sinergia              |

---

## 🧑‍🤝‍🧑 Estilo coletivo por time

* **T1** — predominância *Consistente*.
* **Weibo Gaming** — *Agressivo*.
* **BLG** — *Agressivo*.
* **Gen.G** — *Consistente*.

---

## 🧪 Observações técnicas

* LabelEncoder para dados categóricos.
* StandardScaler (z-score).
* Train/test split estratificado.
* StratifiedKFold (5 folds).
* **MLPClassifier** com `(128, 64)`, ReLU, `max_iter=3000`, `random_state=42`.

---

## 👩‍💻 Autoria e Créditos

* **Autora:** Laura Barbosa Henrique (`@tinywin`)
* **Instituição:** Universidade Federal do Tocantins (UFT)
* **Disciplina:** Inteligência Artificial — 2025/02
* **Docente:** Prof. Dr. Alexandre Rossini
* **Contato:** `laura.henrique@mail.uft.edu.br`

**Dataset:**
["2024 LoL Championship Player Stats and Swiss Stage"](https://www.kaggle.com/datasets/anmatngu/2024-lol-championship-player-stats-and-swiss-stage)
Autor: **nxank4 (An)** — DOI: [10.34740/kaggle/dsv/9722676](https://doi.org/10.34740/kaggle/dsv/9722676)

---

## ⚖️ Licença e Uso

Projeto **educacional**, sem fins comerciais.
Código e experimentos liberados para **aprendizado e pesquisa**, respeitando os termos do Kaggle.

---

## 🧾 Resumo simples

> “Treinei uma rede neural para reconhecer o estilo de jogo de jogadores do Mundial de LoL 2024 usando estatísticas reais.
> A IA aprendeu a identificar perfis como Agressivo, Carregador e Visionário, alcançando cerca de **70% de acerto**.
> Mesmo com boas previsões, o modelo mostra que números nem sempre capturam o fator humano — por isso, a T1 continua sendo a campeã real.”

---

## 🏁 Conclusão

A rede MLP identificou **padrões estatísticos coerentes** com estilos reais.
Acertou 3 dos 4 times do Top 4 e destacou limitações quantitativas — sem captar aspectos humanos como:

* sinergia em série MD5,
* adaptação de draft,
* leitura tática,
* controle emocional,
* impacto do MVP.

---

Deseja que eu gere uma **versão PDF formatada** com este layout (capa, seções, ícones e sumário automático)?
