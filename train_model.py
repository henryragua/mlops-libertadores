import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import argparse

def main(fit_intercept: bool, registered_model_name: str):
    # Cargar dataset de diabetes
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Parámetros
    params = {
        "fit_intercept": fit_intercept
    }

    # Entrenar modelo
    model = LinearRegression(**params)
    model.fit(X_train, y_train)

    # Predicciones y métricas
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Regresion Lineal - Diabetes")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = pd.DataFrame(X_train).head(5)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="linear_model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name
        )

        mlflow.set_tag("modelo", "regresion_lineal")
        mlflow.set_tag("dataset", "diabetes sklearn")
        mlflow.set_tag("autor", "estudiante")

        print(f"✅ Modelo registrado como '{registered_model_name}'. MSE: {mse:.4f}, R²: {r2:.4f}")

# --------------------------
# EJECUCIÓN DESDE CONSOLA
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento y registro de modelo de regresión lineal con MLflow")
    parser.add_argument("--fit_intercept", type=bool, default=True, help="Usar o no intercepto en el modelo")
    parser.add_argument("--model_name", type=str, default="regresion-lineal-diabetes", help="Nombre del modelo en el MLflow Model Registry")
    args = parser.parse_args()

    main(fit_intercept=args.fit_intercept, registered_model_name=args.model_name)
