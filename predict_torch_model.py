import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import argparse

def main(model_uri: str, input_csv: str, output_csv: str = None):
    # -----------------------
    # Configurar Tracking URI
    # -----------------------
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # -----------------------
    # Cargar el modelo
    # -----------------------
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    print(f"✅ Modelo cargado desde: {model_uri}")

    # -----------------------
    # Cargar los datos
    # -----------------------
    data = pd.read_csv(input_csv, header=None)
    print(f"📊 Nuevos datos cargados desde {input_csv}")
    print(data.head())

    # Convertir a tensor
    data_tensor = torch.tensor(data.values, dtype=torch.float32)

    # -----------------------
    # Realizar predicciones
    # -----------------------
    with torch.no_grad():
        predictions = model(data_tensor).numpy().flatten()

    data["prediction"] = predictions

    print("\n🔮 Predicciones:")
    print(data[["prediction"]])

    # -----------------------
    # Guardar si se indicó
    # -----------------------
    if output_csv:
        data.to_csv(output_csv, index=False)
        print(f"\n📁 Resultados guardados en: {output_csv}")

# -----------------------
# Ejecución desde consola
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Usar modelo PyTorch de MLflow para hacer predicciones")
    parser.add_argument("--model-uri", type=str, required=True, help="Ruta del modelo registrado (ej: models:/torch-diabetes-regressor/1)")
    parser.add_argument("--input-csv", type=str, required=True, help="Ruta al archivo CSV de entrada")
    parser.add_argument("--output-csv", type=str, help="(Opcional) Ruta para guardar las predicciones")

    args = parser.parse_args()

    main(args.model_uri, args.input_csv, args.output_csv)
