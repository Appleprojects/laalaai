from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client

app = FastAPI()

# Define the request model
class PredictionRequest(BaseModel):
    input_text: str
    action: str

# Define the response model
class PredictionResponse(BaseModel):
    output_text: str
    output_audio: dict

# Initialize Gradio client
client = Client("Harishwar/lalai")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        result = client.predict(
            input_text=request.input_text,
            action=request.action,
            api_name="/predict"
        )
        return PredictionResponse(output_text=result[0], output_audio=result[1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
