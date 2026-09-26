# Detecção de Fraudes em Cartões de Crédito

Este projeto consiste numa pipeline modular e interativa em Python (CLI) para identificação de transações fraudulentas em cartões de crédito, utilizando algoritmos de Machine Learning otimizados para dados altamente desbalanceados.

---

## Contexto e Problema

O principal desafio na detecção de fraudes financeiras reside no **extremo desbalanceamento dos dados**: as transações legítimas representam a grande maioria dos eventos, enquanto as fraudes correspondem a uma fração inferior a 1% do total.

### Por que o desbalanceamento altera a avaliação?
Avaliar um modelo de fraude utilizando apenas a métrica de **Acurácia** é um erro grave. Um modelo ingênuo que classifique todas as transações como legítimas alcançaria cerca de 99,8% de acurácia, mas falharia em identificar 100% das fraudes. 

Por essa razão, o foco da avaliação está nas seguintes métricas da classe minoritária (Fraude):
* **Recall (Sensibilidade):** Mede a capacidade do modelo em capturar o maior número possível de fraudes reais (minimizar Falsos Negativos).
* **Precisão:** Mede a proporção de alertas emitidos pelo modelo que realmente correspondem a fraudes reais (minimizar Falsos Positivos).
* **F1-Score:** A média harmônica entre Precisão e Recall, garantindo um equilíbrio operacional realista.

---

## Preparação dos Dados

A etapa de engenharia de recursos e pré-processamento seguiu os seguintes passos:

1. **Carregamento Dinâmico:** O dataset é carregado diretamente via link externo durante a execução, mantendo o repositório leve e sem o armazenamento direto de dados pesados.
2. **Escalonamento e Normalização:** Aplicação de `RobustScaler` / `StandardScaler` nas variáveis de valor (`Amount`) e tempo (`Time`) para neutralizar o impacto de *outliers*.
3. **Divisão Estratificada:** Divisão dos dados em treino e teste utilizando `train_test_split` com o parâmetro `stratify=y`, garantindo que a proporção original de fraudes seja preservada em todas as partições.
4. **Tratamento do Desbalanceamento:** Aplicação de técnicas de reamostragem (como SMOTE / Class Weights) no conjunto de treino para evitar o viés do algoritmo em direção à classe majoritária.

---

## Comparação entre Modelos

Abaixo consta a comparação das métricas obtidas na classe de **Fraude (Classe 1)** entre as abordagens testadas:

| Modelo | Precisão (Classe 1) | Recall (Classe 1) | F1-Score (Classe 1) |
| :--- | :---: | :---: | :---: |
| **Regressão Logística (Baseline)** | ~0.85 | ~0.62 | ~0.72 |
| **Random Forest** | ~0.92 | ~0.78 | ~0.84 |
| **XGBoost (Modelo Final)** | **~0.94** | **~0.83** | **~0.88** |

---

## Limiar de Decisão e Explicabilidade (SHAP)

### Limiar de Decisão (*Decision Threshold*)
O limiar padrão de classificação ($0.5$) foi ajustado para **$0.30$**. 
* **Justificativa do negócio:** No contexto bancário, o custo de não detectar uma fraude (Falso Negativo) é significativamente maior do que o custo de verificar uma transação legítima suspeita (Falso Positivo). A redução do limiar aumentou o **Recall** do modelo sem degradar excessivamente a **Precisão**.

### Interpretação com SHAP (*SHapley Additive exPlanations*)
A análise de explicabilidade via SHAP mostrou que:
* As variáveis anônimas **V14**, **V10**, **V12** e **V17** possuem o maior impacto no aumento da probabilidade de fraude.
* Valores extremamente baixos nestas variáveis específicas correlacionam-se fortemente com comportamento fraudulento.
* A variável `Amount` apresenta impacto secundário, atuando principalmente em conjunto com as variáveis de comportamento.

---

## 💡 Diferenciais em Relação à Solução de Referência (Expert)

Em comparação com a abordagem baseline inicial da demonstração, foram implementadas as seguintes melhorias técnicas e arquiteturais:

1. **Arquitetura Modular CLI:** Transformação do script único em módulos reutilizáveis (`train.py`, `predict_csv.py`, `main.py`) acionados por um menu interativo no terminal.
2. **Otimização do Limiar:** Troca do limiar padrão estático por uma variação otimizada com base nas curvas de Precision-Recall.
3. **Persistência e Reprodutibilidade:** Exportação automática da Matriz de Confusão para imagem e serialização do modelo treinado (`.pkl`) em diretórios dedicados e protegidos pelo `.gitignore`.
4. **Tratamento de Exceções na Inferência em Lote:** Módulo `predict_csv.py` robusto, garantindo o retorno ao menu principal em caso de falha de leitura de arquivos sem derrubar a aplicação.

---

## 📂 Estrutura do Repositório

```text
├── predict_results/         # Resultados de predições exportados em CSV
│   └── .gitkeep
├── train_saves/             # Artefatos do treino (Matriz de Confusão e modelo .pkl)
│   └── .gitkeep
├── .gitignore               # Regras de exclusão de arquivos temporários e binários
├── amostra_teste.csv        # Dataset de teste em lote para execução local
├── main.py                  # CLI Controller e menu interativo do sistema
├── predict_csv.py           # Módulo de inferência e predição em lote
├── requirements.txt         # Lista de dependências e bibliotecas do projeto
└── train.py                 # Módulo de treinamento, avaliação e persistência
