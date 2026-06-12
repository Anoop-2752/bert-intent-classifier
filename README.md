# BERT Intent Classifier

Fine-tuned `bert-base-uncased` on the ATIS dataset for airline travel intent classification.

- 26 intent classes
- 97.2% accuracy on test set
- Served via FastAPI REST API

## Model
https://huggingface.co/Anoop2752/bert-intent-classifier

## Run locally
pip install -r requirements.txt
uvicorn app.main:app --reload

## API
POST /predict
{"text": "i want to fly from boston to denver"}

## Tech Stack
HuggingFace Transformers · BERT · FastAPI · Docker