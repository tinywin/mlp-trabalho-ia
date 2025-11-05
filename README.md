# 🎮 MLP – Estilos de Jogo no LoL Worlds 2024

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-MLPClassifier-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-data%20analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-graphs-informational)](https://matplotlib.org/)
[![Kaggle Dataset](https://img.shields.io/badge/dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/anmatngu/2024-lol-championship-player-stats-and-swiss-stage)

Este projeto treina uma rede neural artificial (MLP) para descobrir e classificar estilos de jogo de jogadores profissionais do League of Legends World Championship 2024.

O modelo usa Inteligência Artificial para analisar estatísticas reais do Worlds 2024 e identificar padrões de estilo de jogo entre profissionais.  
Além de classificar jogadores, a IA também calcula a **sinergia de cada time** (combinação de estilos + desempenho médio) para prever o **Top 4 técnico** e um **MVP IA**.

## 📚 Sumário rápido
➡️ [Como foi feito](#-o-que-foi-feito-explicação-simples)  
➡️ [Estilos criados](#-os-estilos-de-jogo-criados)  
➡️ [Critérios de classificação](#-critérios-de-classificação-por-estilo)  
➡️ [Sinergia de time e Top 4 IA](#-sinergia-de-time-e-campeão-ia)  
➡️ [Como rodar o projeto](#️-como-usar)  
➡️ [Resultados e gráficos](#-entendendo-os-resultados)  
➡️ [Créditos e licença](#-autoria-e-créditos)

---

## 🧠 O que foi feito (explicação simples)

1. Foram coletados dados reais de **81 jogadores** do campeonato  
   (fonte: [Kaggle Dataset](https://www.kaggle.com/datasets/anmatngu/2024-lol-championship-player-stats-and-swiss-stage)).
2. As estatísticas foram tratadas (remoção de %, vírgulas, NaNs) e **padronizadas** para que a IA consiga aprender com números em escalas comparáveis.
3. Foram criados **rótulos de estilo de jogo** baseados no desempenho dos jogadores — estilos como “Agressivo”, “Carregador”, “Visionário” etc.
4. Foi treinada uma **MLP (Multi-Layer Perceptron)** do Scikit-learn, que aprendeu a mapear estatísticas → estilo de jogo primário.
5. O modelo foi avaliado com:
   - **Hold-out** (treino/teste com split estratificado)
   - **Validação cruzada (5-fold)** usando pipeline (StandardScaler + MLP).
6. Foram gerados **gráficos e relatórios automáticos** para visualizar o desempenho, estilos e sinergia por time.

---

## 🔍 O que é uma MLP?

A **MLP (Multi-Layer Perceptron)** é um tipo de rede neural artificial totalmente conectada que aprende padrões a partir de exemplos.

Neste projeto, a MLP recebe como entrada estatísticas dos jogadores (por exemplo: **KDA, DPM, GPM, KP%, visão, Solo Kills, GD@15**) e aprende a associar esses números a um **estilo de jogo primário**.

---

## 🧩 Os estilos de jogo criados

Estilos definidos para representar como um jogador tende a atuar:

| Estilo | Explicação simples |
| --- | --- |
| 🗡️ **Agressivo** | Parte pra cima, busca abates e pressiona o mapa. |
| 💪 **Carregador** | Principal fonte de dano e vitórias do time (carry). |
| 🧱 **Consistente** | Joga de forma segura, erra pouco, mantém bom desempenho. |
| ⚔️ **Duelista** | Forte em lutas 1x1, depende da mecânica individual. |
| ⚖️ **Equilibrado** | Mistura ataque e defesa, joga de forma adaptável. |
| 💥 **Volátil** | Instável: pode jogar muito bem ou muito mal (imprevisível). |
| 🩹 **Suporte** | Ajuda o time com visão, cura, proteção e controle. |
| 🔮 **Visionário** | Foca em controle de mapa, leitura tática e visão estratégica. |

Esses estilos foram **criandos via regras heurísticas** a partir das estatísticas da base, inspirados no comportamento de jogadores profissionais.

Cada jogador pode receber **múltiplos estilos** (multiestilo), refletindo perfis híbridos (ex.: *Carregador + Duelista*).  
Para treinar a MLP, é escolhido um **Estilo Primário**, mas a análise completa mantém todos os estilos associados.

---

## 🧩 Critérios de Classificação por Estilo

Os termos “média” e “percentil 75 (p75)” são calculados sobre toda a base de jogadores.

| Estilo        | Regra (simplificada)                                           | Interpretação breve |
|---------------|----------------------------------------------------------------|---------------------|
| 💪 **Carregador** | DPM > p75 ∧ GPM > média ∧ KDA > média                          | Dano alto, bom ouro e poucas mortes — “carrega” o time |
| 🗡️ **Agressivo**  | DPM > média ∧ (KP% > média ∨ Solo Kills > média)               | Foco em dano e presença em abates; cria jogadas |
| 🔮 **Visionário** | VSPM > média ∧ WPM > média ∧ DPM < média                       | Prioriza visão/controle de mapa, não dano |
| 🩹 **Suporte**    | KP% > média ∧ WPM > média ∧ GPM < média                        | Alta participação e visão com pouco ouro |
| 🧱 **Consistente**| KDA > p75 ∧ Avg Deaths < média                                  | Estável, difícil de punir |
| 💥 **Volátil**    | GD@15 < 0 ∧ Avg Deaths > média                                  | Early negativo e mortes acima da média; desempenho oscilante |
| ⚔️ **Duelista**   | Solo Kills > p75 ∧ DPM > média                                 | Mecânica forte e confiança no 1x1 |
| ⚖️ **Equilibrado**| Nenhuma das regras acima                                        | Neutro, estável, sem extremos — meio-termo |

**Prioridade do Estilo Primário (para treino):**  
Carregador > Agressivo > Visionário > Suporte > Consistente > Volátil > Duelista > Equilibrado.

---

## 🤝 Sinergia de Time e Campeão IA

Além dos estilos individuais, o projeto calcula um **índice de sinergia por equipe**, combinando:

- **Diversidade e cobertura de estilos core**  
  (Carregador, Agressivo, Visionário, Suporte, Consistente, Duelista)
- **Desempenho médio do time**:  
  - KDA médio  
  - DPM médio  
  - GD@15 médio  

A sinergia é calculada como:

> **Synergy Score = 0,7 · StyleScore + 0,3 · PerfScore**

Com base nisso, a IA gera:

- 🥇 o **“campeão IA”** (time mais completo em estilos + performance),
- 🥈 o **vice técnico**,  
- 🏅 o **MVP IA** (jogador mais impactante dentro do time campeão, combinando z-score de DPM, KDA, KP% + bônus por estilo Carregador/Agressivo).

Na execução de referência do projeto:

- **Top 4 IA (sinergia estilo + performance)**  
  1. **Weibo Gaming** — sinergia ≈ 4,07  
  2. **T1** — sinergia ≈ 4,06  
  3. **Gen.G** — sinergia ≈ 3,62  
  4. **Team Liquid** — sinergia ≈ 3,49  

- **MVP IA:** `xiaohu` (Weibo Gaming) — *Carregador, Agressivo, Consistente*  

Já o **Top 4 real do Worlds 2024** foi:

- 🥇 **T1** (campeã)  
- 🥈 **Bilibili Gaming (BLG)**  
- 🥉–4 **Weibo Gaming (WBG)** e **Gen.G**, ambos em 3–4 (não há disputa de 3º lugar)  

Ou seja, a IA acerta **3 dos 4 times do Top 4 real** (T1, Weibo, Gen.G), apenas substituindo a **BLG por Team Liquid** nas previsões, o que é um resultado bem interessante dado que o modelo vê só estatísticas agregadas.

---

## ⚙️ Como usar

1. Instale as dependências:

```powershell
pip install -r requirements.txt
````

2. Execute o script principal:

```powershell
python .\src\mlp_estilo_lol_final.py
```

O programa:

* lê a base de dados,
* calcula os estilos multiestilo e o estilo primário,
* treina a rede neural,
* avalia o modelo,
* gera relatórios e gráficos na pasta `outputs/`.

---

## 🗂️ Estrutura do Projeto

```text
📁 src/
 ├── mlp_estilo_lol_final.py              # Script principal
 ├── player_statistics_cleaned_final.csv  # Base de dados com ~81 jogadores
📁 outputs/
 ├── confusion_matrix_estilo_*.png        # Matriz de confusão
 ├── estilos_bar_*.png                    # Quantos jogadores de cada estilo (previstos)
 ├── estilos_pie_*.png                    # Proporção de estilos (previstos)
 ├── estilos_multi_bar_*.png              # Distribuição multiestilo (agregado)
 ├── estilos_multi_pie_*.png              # Proporção multiestilo (agregado)
 ├── relatorio_estilos_*.txt              # Relatório completo de resultados
 └── predicoes_completas_*.csv            # Tabela com previsões detalhadas
```

---

## 📈 Entendendo os resultados

### ✅ Métricas da versão atual

Na execução de referência (relatório colado acima), o modelo obteve:

* **Acurácia (hold-out):** 0,68 → **68%**
* **Precisão ponderada:** ≈ 0,73
* **Recall ponderado:** ≈ 0,68
* **F1 ponderado:** ≈ 0,69

E na validação cruzada (5 folds):

* **Acurácia média (CV 5-fold):** ≈ 0,74
* **Desvio padrão:** ≈ 0,06

Isso significa, em linguagem simples, que o modelo acerta **em torno de 70%** dos estilos primários, em um problema com **múltiplas classes** e rótulos heurísticos.

### 🧾 O que significam as métricas?

| Métrica      | O que significa                                                                      |
| ------------ | ------------------------------------------------------------------------------------ |
| **Acurácia** | Proporção de previsões totais que a IA acertou.                                      |
| **Precisão** | Quando a IA diz que um jogador é de um estilo, o quanto isso costuma estar correto.  |
| **Recall**   | Dos jogadores que realmente têm aquele estilo, quantos a IA consegue identificar.    |
| **F1-Score** | Equilíbrio entre precisão e recall (quanto maior, melhor o compromisso entre ambos). |

Exemplo: acurácia de **~70%** significa que a IA acerta cerca de **7 a cada 10 jogadores** na classificação do estilo primário.

### 🧾 Interpretação do relatório

O relatório gera:

* Distribuição de estilos **multiestilo** (todas as tags aplicadas aos jogadores).
* Distribuição do **Estilo Primário** usado no treino.
* Matriz de confusão mostrando em quais estilos a IA mais erra/confunde.
* Top 4 de sinergia de time segundo a IA.
* Campeão IA, vice técnico e MVP IA.
* Um resumo textual do **estilo coletivo por time** (predominância: Agressivo, Consistente, Volátil, etc.).

> 💡 **Desequilíbrio de classes**
> Estilos com pouquíssimos exemplos (como Suporte e Duelista) tendem a ter métricas fracas (por exemplo, F1 ≈ 0 em algumas execuções), simplesmente por falta de dados suficientes.
> Com mais jogadores rotulados nesses estilos ou técnicas de balanceamento (oversampling/SMOTE, por exemplo), o modelo pode melhorar nesses casos específicos.

---

## 🖼️ Interpretação dos gráficos

| Arquivo                         | O que mostra                      | Como interpretar                                        |
| ------------------------------- | --------------------------------- | ------------------------------------------------------- |
| `confusion_matrix_estilo_*.png` | Matriz de confusão                | Acertos na diagonal; erros nas células fora da diagonal |
| `estilos_bar_*.png`             | Distribuição de estilos (barras)  | Quantos jogadores em cada classe prevista               |
| `estilos_pie_*.png`             | Proporção de estilos (pizza)      | Percentual de cada classe prevista                      |
| `estilos_multi_bar_*.png`       | Distribuição multiestilo (barras) | Frequência dos estilos considerando todas as tags       |
| `estilos_multi_pie_*.png`       | Proporção multiestilo (pizza)     | Percentual de aparição de cada estilo (multiestilo)     |
| `predicoes_completas_*.csv`     | Tabela detalhada de previsões     | Estilos previstos por jogador                           |
| `relatorio_estilos_*.txt`       | Relatório completo                | Métricas gerais, destaques e notas                      |

---

## 🧑‍🤝‍🧑 Estilo coletivo por time

O relatório também traz um **resumo textual por equipe**, gerado com base nos estilos individuais dos jogadores. Exemplos (da execução de referência):

* **T1**: predominância **“Consistente”** (um Carregador principal, tendência agressiva, boa presença de visão/suporte, núcleo estável)
* **Weibo Gaming**: predominância **“Agressivo”** (foco em Carregadores, alta pressão de mapa, boa visão)
* **BLG**: predominância **“Agressivo”** (dupla de carries forte, suporte visionário, time explosivo)
* **Gen.G**: predominância **“Consistente”** (foco em Carregadores, núcleo muito estável, boa leitura de mapa)

---

## 🧪 Observações técnicas

* Dados categóricos convertidos via **LabelEncoder** (ex.: `Country`, `FlashKeybind`).
* Dados numéricos padronizados com **StandardScaler** (z-score).
* Split estratificado com **train_test_split** (hold-out).
* Validação adicional com **StratifiedKFold (5 folds)** e `Pipeline(StandardScaler + MLPClassifier)`.
* Modelo principal:

  * `MLPClassifier`
  * Camadas ocultas: `(128, 64)`
  * Ativação: **ReLU**
  * `max_iter = 3000`
  * `random_state = 42`

---

## 👩‍💻 Autoria e Créditos

* **Autora:** Laura Barbosa Henrique (`@tinywin`)
* **Instituição:** Universidade Federal do Tocantins (UFT)
* **Disciplina:** Inteligência Artificial — 2025/02
* **Docente:** Prof. Dr. Alexandre Rossini
* **Contato:** `laura.henrique@mail.uft.edu.br`

**Dataset original:**

> ["2024 LoL Championship Player Stats and Swiss Stage"](https://www.kaggle.com/datasets/anmatngu/2024-lol-championship-player-stats-and-swiss-stage)
> Autor: **nxank4 (An)** — DOI: [10.34740/kaggle/dsv/9722676](https://doi.org/10.34740/kaggle/dsv/9722676)

---

## ⚖️ Licença e Uso

Este projeto é **educacional e sem fins comerciais**.
O código e os experimentos são disponibilizados para fins de **aprendizado e pesquisa acadêmica**, respeitando:

* direitos autorais do dataset original, e
* termos de uso da plataforma Kaggle.

---

## 🧾 Resumo simples

> “Treinei uma rede neural para reconhecer o estilo de jogo de jogadores do Mundial de LoL 2024 usando estatísticas reais.
> A IA aprendeu a identificar perfis como Agressivo, Carregador e Visionário, alcançando cerca de **70% de acerto** (≈68% no teste hold-out e ≈74% em validação cruzada).
> Mesmo com boas previsões, o modelo mostra que números nem sempre capturam o fator humano — por isso, a T1 continua sendo a campeã real.”

---

## 🏁 Conclusão

A rede MLP identificou **padrões estatísticos coerentes** com estilos observáveis nos profissionais.
Além de classificar corretamente o perfil de vários jogadores, a IA também produziu um **Top 4 técnico** muito próximo do resultado real, acertando 3 dos 4 times que chegaram ao Top 4 do torneio (T1, Weibo e Gen.G).

Ainda assim, o modelo expõe limites naturais de abordagens puramente quantitativas. Ao considerar apenas métricas agregadas, a IA tende a privilegiar consistência numérica; com isso, times como a Weibo podem aparecer mais “equilibrados” nos dados, enquanto a **T1** venceu por fatores qualitativos:

* sinergia em série MD5,
* adaptação de draft,
* leitura tática em tempo real,
* controle emocional em jogos decisivos,
* e, claro, o impacto do MVP.