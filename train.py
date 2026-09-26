import os
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def run_training():
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
        n_estimators=100,  
        scale_pos_weight=ratio,  
        max_depth=6,  
        learning_rate=0.1,  
        random_state=42,  
        n_jobs=-1, 
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

    pasta_treino = "train_saves"

    # Cria a pasta caso ela ainda não exista
    os.makedirs(pasta_treino, exist_ok=True)

    plt.title("Matriz de Confusão - XGBoost (Limiar 0.30)")

    # Salvar o gráfico da Matriz de Confusão no diretório de treino
    caminho_grafico = os.path.join(pasta_treino, "matriz_confusao.png")
    plt.savefig(caminho_grafico, dpi=300, bbox_inches="tight")
    print("📊 Gráfico da Matriz de Confusão salvo como 'matriz_confusao.png'!")
    plt.close()  
    print(f"📊 Gráfico salvo com sucesso em: '{caminho_grafico}'")

    # Salvar o modelo treinado na respectiva pasta.
    caminho_modelo = os.path.join(pasta_treino, "modelo_antifraude_xgb.pkl")
    joblib.dump(model, caminho_modelo)
    print(f"✅ Modelo salvo com sucesso em: '{caminho_modelo}'")