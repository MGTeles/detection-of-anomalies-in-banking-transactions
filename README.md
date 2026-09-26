# Sistema de Detecção de Anomalias em Cartões de Crédito (XGBoost)


---

## Visão Geral e Problema de Negócio

Em sistemas de detecção de fraude, o custo de um **Falso Negativo** (não detectar uma fraude real) costuma ser dramaticamente superior ao custo de um **Falso Positivo** (bloquear temporariamente uma transação legítima). 

Para responder a esse desafio, este projeto aplica:
1. **Ponderação de Classes (`scale_pos_weight`)**: Compensação estatística para a raridade dos casos de fraude.
2. **Ajuste do Limiar de Decisão (0.30)**: Redução do limiar padrão de 50% para 30%, elevando a sensibilidade (*Recall*) do modelo e capturando uma proporção maior de fraudes sem comprometer excessivamente a precisão.

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