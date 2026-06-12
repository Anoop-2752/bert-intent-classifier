from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="BERT Intent Classifier")

# Load your model directly from HuggingFace Hub
classifier = pipeline(
    "text-classification",
    model="Anoop2752/bert-intent-classifier"
)

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "BERT Intent Classifier is running"}

@app.post("/predict")
def predict(input: TextInput):
    result = classifier(input.text)[0]
    return {
        "text"      : input.text,
        "intent"    : result["label"],
        "confidence": round(result["score"], 4)
    }