import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# CARREGAMENTO DOS DADOS
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# SEPARAÇÃO DE RECURSOS (X) E ALVO (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# DIVISÃO EM DADOS DE TREINO E TESTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# CÁLCULO DO PESO DE DESBALANCEAMENTO PARA O XGBOOST
ratio = (len(y_train) - sum(y_train)) / sum(y_train)

# INSTANCIAÇÃO E CONFIGURAÇÃO DO XGBOOST
model = XGBClassifier(
    n_estimators=100,  # Número de árvores
    scale_pos_weight=ratio,  # Equivalente ao class_weight="balanced"
    max_depth=6,  # Profundidade máxima de cada árvore
    learning_rate=0.1,  # Taxa de aprendizado
    random_state=42,  # Reproduzibilidade
    n_jobs=-1,  # Usa todas as CPUs
)

# TREINAMENTO DO MODELO
model.fit(X_train, y_train)

# OTIMIZAÇÃO: THRESHOLD TUNING (LIMIAR DE DECISÃO DE 30%)
y_probabilities = model.predict_proba(X_test)[:, 1]

novo_threshold = 0.30
y_pred_ajustado = (y_probabilities >= novo_threshold).astype(int)

# AVALIAÇÃO DOS RESULTADOS E VISUALIZAÇÃO GRÁFICA
print("--- RELATÓRIO DE DESEMPENHO (XGBOOST - LIMIAR 30%) ---")
print(classification_report(y_test, y_pred_ajustado))

# VISUALIZAÇÃO DA MATRIZ DE CONFUSÃO
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_ajustado, display_labels=["Legítima", "Fraude"], cmap="Blues"
)
plt.title("Matriz de Confusão - XGBoost (Limiar 0.30)")
plt.show()

# SALVANDO O MODELO TREINADO EM UM ARQUIVO .pkl
joblib.dump(model, "modelo_antifraude_xgb.pkl")
print("Modelo salvo com sucesso em 'modelo_antifraude_xgb.pkl'!")