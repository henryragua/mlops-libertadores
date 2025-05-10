from huggingface_hub import HfApi
import os

# Variables (ajusta estos valores)
HF_REPO_ID = "henryragua/mlops-libertadores"  # Tu repo en Hugging Face
MODEL_PATH = "models/prediccion-libertadores.pkl"       # Ruta de tu modelo local

# Autenticación con token desde GitHub Actions
api = HfApi(token=os.getenv("HF_TOKEN"))

# Subir el modelo
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="modelo.pkl",
    repo_id=HF_REPO_ID,
    repo_type="model",
    commit_message="Subida automática del modelo desde GitHub Actions"
)

print("✅ Modelo subido exitosamente a Hugging Face.")
