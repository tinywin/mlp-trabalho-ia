# 🎮 MLP – Estilos de Jogo no LoL Worlds 2024

Este projeto treina uma rede neural artificial (MLP) para descobrir e classificar estilos de jogo de jogadores profissionais do League of Legends World Championship 2024.

Este projeto usa Inteligência Artificial para analisar estatísticas reais do League of Legends Worlds 2024 e identificar padrões de estilo de jogo entre os profissionais.
A rede neural aprende a diferenciar perfis como Agressivo, Carregador e Visionário com base em métricas de dano, visão e consistência.
O objetivo é demonstrar como técnicas de Machine Learning podem apoiar a compreensão de comportamento e desempenho em esportes eletrônicos.

## 📚 Sumário rápido
➡️ [Como foi feito](#-o-que-foi-feito-explicação-simples)  
➡️ [Estilos criados](#-os-estilos-de-jogo-criados)  
➡️ [Como rodar o projeto](#️-como-usar)  
➡️ [Resultados e gráficos](#-entendendo-os-resultados)  
➡️ [Créditos e licença](#-autoria-e-créditos)

## 🧠 O que foi feito (explicação simples)

1. Coletamos dados reais de 81 jogadores do campeonato (fonte: [Kaggle Dataset](https://www.kaggle.com/datasets/anmatngu/2024-lol-championship-player-stats-and-swiss-stage)).
2. Tratamos e padronizamos os números (ex.: converter “país” e “atalho do flash” em valores que a IA entende).
3. Criamos rótulos de estilo baseados no desempenho dos jogadores — estilos como “Agressivo” ou “Visionário”.
4. Treinamos uma MLP (Rede Neural Multicamadas) do Scikit-learn, que aprendeu a relacionar estatísticas → estilo de jogo.
5. Avaliamos o modelo com métricas de acerto (acurácia, precisão, F1, etc.).
6. Geramos gráficos e relatórios automáticos para visualizar os resultados.

## 🔍 O que é uma MLP?

A MLP (Multi-Layer Perceptron) é um tipo de rede neural artificial que aprende padrões nos dados. No projeto, ela recebe números sobre cada jogador (KDA, dano, ouro, visão etc.) e aprende a reconhecer perfis de jogo.

## 🧩 Os estilos de jogo criados

Estilos definidos para representar como um jogador tende a atuar:

| Estilo | Explicação simples |
| --- | --- |
| 🗡️ Agressivo | Parte pra cima, busca abates e pressiona o mapa. |
| 💪 Carregador | Principal fonte de dano e vitórias do time (carry). |
| 🧱 Consistente | Joga de forma segura, erra pouco, mantém bom desempenho. |
| ⚔️ Duelista | Forte em lutas 1x1, depende da mecânica individual. |
| ⚖️ Equilibrado | Mistura ataque e defesa, joga de forma adaptável. |
| 💥 Pipoqueiro | Instável: pode jogar muito bem ou muito mal (imprevisível). |
| 🩹 Suporte | Ajuda o time com visão, cura, proteção e controle. |
| 🔮 Visionário | Foca em controle de mapa, leitura tática e visão estratégica. |

Esses estilos foram criados a partir das métricas da base e inspirados no comportamento de jogadores profissionais. A IA aprende a associar estatísticas (números) a esses rótulos.

## ⚙️ Como usar

1. Instale as dependências:

	```powershell
	pip install -r requirements.txt
	```

2. Execute o script principal:

	```powershell
	python .\src\mlp_estilo_lol_final.py
	```

O programa lê a base de dados, treina a rede neural e gera automaticamente:

- relatórios em texto (no terminal e em `outputs/`)
- imagens dos gráficos e métricas do modelo

## 🗂️ Estrutura do Projeto

```
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

## 📈 Entendendo os resultados

### ✅ Métricas

| Métrica | O que significa |
| --- | --- |
| Acurácia | Quantas previsões totais a IA acertou. |
| Precisão | Quando a IA diz que é um estilo, o quanto ela acerta. |
| Recall | Dos exemplos realmente positivos, quantos foram detectados. |
| F1-Score | Combinação entre precisão e recall (quanto mais equilibrado, melhor). |

Exemplo: acurácia de ~80% significa que a IA acerta 8 a cada 10 jogadores.

### 🧾 Interpretação do relatório

A saída mostra:

- Quantos jogadores por estilo (previstos) e multiestilos agregados.
- Quais times têm predominância de qual estilo.
- Quem foi o MVP da IA (com base em desempenho e estilos).
- Matriz de confusão (onde a IA confunde um estilo com outro).

Observação: mesmo que a IA tenha escolhido Weibo Gaming como o time mais equilibrado, na vida real a T1 foi campeã — estatísticas nem sempre capturam fatores humanos (adaptação, estratégia, pressão).

Nota sobre desequilíbrio de classes: estilos com poucos exemplos (como Suporte e Duelista) tendem a apresentar métricas mais baixas (até F1 ≈ 0) por falta de dados suficientes. Em cenários assim, técnicas de balanceamento (ex.: oversampling/SMOTE) ou coleta de mais exemplos ajudam a melhorar o aprendizado nessas classes raras.

### � Exemplo de saída do relatório

```
Acurácia: 0.80
Precisão média: 0.86
Estilo mais comum previsto: Agressivo
Time mais equilibrado: Weibo Gaming
MVP segundo a IA: xiaohu (Weibo Gaming)
Campeão real: T1 🏆
```

## 🖼️ Interpretação dos gráficos

| Arquivo                         | O que mostra                            | Como interpretar                                   |
|---------------------------------|-----------------------------------------|----------------------------------------------------|
| `confusion_matrix_estilo_*.png` | Matriz de confusão                      | Acertos na diagonal; erros nas células fora da diagonal |
| `estilos_bar_*.png`             | Distribuição de estilos (barras)        | Quantos jogadores em cada classe prevista          |
| `estilos_pie_*.png`             | Proporção de estilos (pizza)            | Percentual de cada classe prevista                 |
| `estilos_multi_bar_*.png`       | Distribuição multiestilo (barras)       | Frequência dos estilos considerando múltiplas tags |
| `estilos_multi_pie_*.png`       | Proporção multiestilo (pizza)           | Percentual de perfis híbridos (multiestilo)        |
| `predicoes_completas_*.csv`     | Tabela detalhada de previsões           | Estilos previstos por jogador                      |
| `relatorio_estilos_*.txt`       | Relatório completo                      | Métricas gerais, destaques e notas                 |

## �🧪 Observações técnicas

- Dados categóricos foram codificados (ex.: LabelEncoder para colunas discretas).
- Dados numéricos foram padronizados (StandardScaler, z-score).
- Validação por Hold-Out; quando aplicável, Cross-Validation pode complementar.
- Modelo: MLPClassifier com camadas (128, 64), ativação ReLU, até 3000 iterações.
- Saídas salvas com timestamps automáticos na pasta `outputs/`.

## 👩‍💻 Autoria e Créditos

Autora: Laura Barbosa Henrique (`tinywin`)

Instituição: Universidade Federal do Tocantins (UFT)

Disciplina: Inteligência Artificial — 2025/02

Docente: Prof. Dr. Alexandre Rossini

Contato: laura.henrique@mail.uft.edu.br

Dataset: ["2024 LoL Championship Player Stats and Swiss Stage"](https://www.kaggle.com/datasets/anmatngu/2024-lol-championship-player-stats-and-swiss-stage) — Autor: **nxank4 (An)** — DOI: [10.34740/kaggle/dsv/9722676](https://doi.org/10.34740/kaggle/dsv/9722676)

## ⚖️ Licença e Uso

Este projeto é educacional e sem fins comerciais. O código e os dados são disponibilizados apenas para aprendizado e pesquisa acadêmica, respeitando os direitos autorais e termos do dataset original.

## 🧾 Resumo simples

Treinei uma rede neural para reconhecer o estilo de jogo de jogadores do Mundial de LoL 2024 usando estatísticas reais.  A IA aprendeu a identificar perfis como Agressivo, Carregador e Visionário, alcançando cerca de 80% de acerto.  
Mesmo com boas previsões, o modelo mostra que números nem sempre capturam o fator humano — por isso, a T1 continua sendo a campeã real.

## 🏁 Conclusão

A rede MLP identificou padrões estatísticos coerentes com estilos observáveis nos profissionais. Apesar de acertos notáveis — como classificar Faker como “Agressivo e Consistente” e Keria como “Visionário e Suporte” — o modelo também expõe limites naturais de abordagens puramente quantitativas. Por considerar apenas estatísticas, a IA privilegia consistência numérica; assim, times como Weibo podem aparecer mais “equilibrados” nos dados, enquanto a T1 venceu por fatores qualitativos (sinergia, leitura tática, adaptação sob pressão) que extrapolam as métricas. Em síntese: redes neurais ajudam a entender desempenho, mas o jogo também depende de decisões humanas, trabalho em equipe e adaptação em tempo real — dimensões que o modelo não captura integralmente.
