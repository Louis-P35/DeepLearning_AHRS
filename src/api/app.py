# Import from trirdparty
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn

# Import from stl
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import from src
from models.model import ModelHandler

# Init a FastAPI container
app: FastAPI = FastAPI()

# Resolve the model path
current_dir: str = os.path.dirname(os.path.abspath(__file__))
model_path: str = os.path.join(current_dir, "models", "artifacts", "sentimentAnalysis_model_large.h5")

# Load the model
print(f"Loading model from {model_path}")
model_handler: ModelHandler = ModelHandler(model_path=model_path)

@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "API is running"}


@app.post("/predict/")
async def predict(data: dict) -> dict:
    """
    Endpoint to make predictions using the loaded model.

    Args:
        data (dict): Dictionary containing the input data.

    Returns:
        dict: Dictionary containing the prediction.
    """
    try:
        input_data = data.get("text", [])  # Get the input text
        if not isinstance(input_data, list):
            input_data = [input_data]  # Ensure it's a list

        prediction = model_handler.predict([input_data])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Start the Uvicorn server to serve the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"text": ["This is a very nice appartment, I loved it!!!", "This is a crappy and dirty flat, do not rent it"]}'
"""



# TODO: Pk fastapi: framework ready pour de la prod, monté en compétence rapide, utilisé par des grosses boites notamment pour faire tourner des modèles d'IA