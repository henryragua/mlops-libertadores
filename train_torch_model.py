import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

import pandas as pd
import numpy as np

# -----------------------------
# Definición de la red neuronal
# -----------------------------
class SimpleRegressor(nn.Module):
    def __init__(self, input_size):
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# -----------------------------
# Función principal del script
# -----------------------------
def main():
    # Cargar datos
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertir a tensores
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Modelo y entrenamiento
    model = SimpleRegressor(input_size=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Entrenamiento básico
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Predicción y métricas
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).numpy().flatten()
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

    # Registrar en MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Torch - Diabetes")

    with mlflow.start_run():
        mlflow.log_param("epochs", 10)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("lr", 0.01)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        input_example = pd.DataFrame(X_train_scaled[:5])
        signature = infer_signature(X_train_scaled, model(torch.tensor(X_train_scaled, dtype=torch.float32)).detach().numpy())

        # Guardar modelo en formato PyTorch y pyfunc
        mlflow.pytorch.log_model(
            model,
            artifact_path="torch_model",
            input_example=input_example,
            signature=signature,
            registered_model_name="torch-diabetes-regressor"
        )

        print(f"✅ Modelo entrenado y registrado. MSE: {mse:.4f}, R²: {r2:.4f}")

# -----------------------------
# Ejecución desde consola
# -----------------------------
if __name__ == "__main__":
    main()
