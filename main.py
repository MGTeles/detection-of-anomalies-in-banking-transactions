import sys
import train
import predict_csv


def menu():
    while True:  # 🔄 Laço que mantém o menu rodando continuamente
        print("\n" + "=" * 50)
        print("      SISTEMA DE DETECÇÃO DE ANOMALIAS (XGBOOST)")
        print("=" * 50)
        print("1. Treinar / Re-treinar o modelo (train.py)")
        print("2. Analisar lote de transações via CSV (predict_csv.py)")
        print("3. Sair")
        print("=" * 50)

        opcao = input("Escolha uma opção (1-3): ").strip()

        if opcao == "1":
            train.run_training()
        elif opcao == "2":
            predict_csv.run_predictions()
        elif opcao == "3":
            print("\nEncerrando o programa... Até logo!")
            sys.exit()  
        else:
            print("\n❌ Opção inválida! Digite um número de 1 a 3.")


if __name__ == "__main__":
    menu()