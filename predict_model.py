import mlflow
import mlflow.sklearn
import pandas as pd
import argparse

def main(model_uri: str, input_csv: str, output_csv: str = None):
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Cargar modelo desde el Model Registry o ruta local
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Modelo cargado desde: {model_uri}")

    # Cargar nuevos datos
    data = pd.read_csv(input_csv, header=None)  # header=None si no hay nombres de columnas
    print(f"📊 Datos cargados desde: {input_csv}")
    print(data.head())

    # Hacer predicciones
    predictions = model.predict(data)
    data["prediction"] = predictions

    print("\n🔮 Predicciones:")
    print(data[["prediction"]])

    # Guardar resultados si se indica
    if output_csv:
        data.to_csv(output_csv, index=False)
        print(f"\n📁 Resultados guardados en: {output_csv}")

# --------------------------
# EJECUCIÓN DESDE CONSOLA
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hacer predicciones usando un modelo registrado en MLflow")
    parser.add_argument("--model-uri", type=str, required=True, help="URI del modelo (ej: models:/regresion-lineal-diabetes/1 o /local/path/to/model)")
    parser.add_argument("--input-csv", type=str, required=True, help="Ruta al archivo CSV con datos de entrada")
    parser.add_argument("--output-csv", type=str, default=None, help="(Opcional) Ruta para guardar las predicciones")
    args = parser.parse_args()

    main(model_uri=args.model_uri, input_csv=args.input_csv, output_csv=args.output_csv)
