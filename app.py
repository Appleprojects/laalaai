from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gradio_client import Client

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define request and response models
class PredictionRequest(BaseModel):
    input_text: str
    action: str

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
