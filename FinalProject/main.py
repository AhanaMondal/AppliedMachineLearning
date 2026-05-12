from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from predict import predict_transaction

app = FastAPI()

EXPECTED_FEATURES = 30

# Define request schema
class Transaction(BaseModel):
    transaction: List[float]

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(data: Transaction):
    transaction = data.transaction

    if len(transaction) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features, got {len(transaction)}"
        )

    return predict_transaction(transaction)