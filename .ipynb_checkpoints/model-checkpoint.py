import automlx
from automlx import Pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

def main():
    # Carrega e prepara os dados (garantindo que sejam DataFrames pandas)
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")  # Convertendo para Series pandas

    # Divisão treino-teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    try:
        # Cria e treina o modelo (sem init)
        print("Treinando o modelo...")
        model = Pipeline(task="classification")  # Versões recentes não precisam de 'engine'
        model.fit(X_train, y_train)

        # Previsões e avaliação
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        print("\nResultados:")
        print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")

        print("\nResumo do modelo:")
        model.print_summary()

    except Exception as e:
        print(f"Erro: {str(e)}")

if __name__ == "__main__":
    main()