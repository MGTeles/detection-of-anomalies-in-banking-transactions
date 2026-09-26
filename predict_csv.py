import os
import sys
import joblib
import pandas as pd

# Carregar o modelo treinado salvo em disco
try:
    model = joblib.load("modelo_antifraude_xgb.pkl")
    print("✅ Modelo carregado com sucesso!")
except FileNotFoundError:
    print(
        "❌ Erro: O arquivo 'modelo_antifraude_xgb.pkl' não foi encontrado na pasta."
    )
    sys.exit()

# Solicitar o caminho do arquivo CSV com as novas transações
caminho_csv = input(
    "Digite o nome ou caminho do arquivo CSV para analisar (ex: novas_transacoes.csv): "
)

try:
    # Lê o CSV enviado pelo usuário
    novos_dados = pd.read_csv(caminho_csv)
    print(f"\n📂 Arquivo carregado com {len(novos_dados)} transações.")
except FileNotFoundError:
    print(f"❌ Erro: O arquivo '{caminho_csv}' não foi encontrado.")
    sys.exit()

# Remover a coluna 'Class' caso ela venha no arquivo de teste
# (Garante que passamos apenas as features para o modelo)
if "Class" in novos_dados.columns:
    X_novos = novos_dados.drop("Class", axis=1)
else:
    X_novos = novos_dados.copy()

# Calcular probabilidades para TODAS as linhas do CSV de uma vez
probabilidades = model.predict_proba(X_novos)[:, 1]

# Aplicar o limiar de decisão (Threshold de 30%)
limiar = 0.30
previsoes = (probabilidades >= limiar).astype(int)

# Adicionar os resultados diretamente ao DataFrame original
novos_dados["Probabilidade_Fraude_%"] = (probabilidades * 100).round(2)
novos_dados["Previsao_Fraude"] = previsoes
novos_dados["Status"] = novos_dados["Previsao_Fraude"].map(
    {0: "Aprovado", 1: "Suspeito/Fraude"}
)

# Exibir um resumo no terminal
total_fraudes = (previsoes == 1).sum()
print("\n" + "=" * 50)
print("              RESUMO PROCESSADO")
print("=" * 50)
print(f"Total de transações analisadas: {len(novos_dados)}")
print(f"Transações Aprovadas:           {len(novos_dados) - total_fraudes}")
print(f"Alertas de Fraude Detectados:  {total_fraudes}")
print("=" * 50)

# Define a pasta de destino
pasta_destino = "predict_results"

# Extrai apenas o nome do arquivo
nome_arquivo_original = os.path.basename(caminho_csv)

# Monta o novo nome com o prefixo 'resultado_'
nome_arquivo_saida = f"resultado_{nome_arquivo_original}"

# Une a pasta com o nome do arquivo para formar o caminho final
caminho_saida_completo = os.path.join(pasta_destino, nome_arquivo_saida)

# Exporta o resultado para dentro da pasta predict_results
novos_dados.to_csv(caminho_saida_completo, index=False)
print(
    f"\n✅ Resultados exportados com sucesso para: '{caminho_saida_completo}'"
)