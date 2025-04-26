from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

# Set the tracking URI for MLFlow
mlflow.set_tracking_uri('http://127.0.0.1:5005')

# Load model

model_uri = 'runs:/ed3f698800614524bed6f680b1bcb613/best_model'


model = mlflow.sklearn.load_model(model_uri)

# Create FastAPI app
app = FastAPI()

# Request body structure (input data)
class CreditData(BaseModel):
    Income: float
    Limit: float
    Cards: float
    Age: float
    Education: float
    Gender: float
    Student: float
    Married: float
    Ethnicity: float
    Balance: float


@app.post("/predict/")
def predict(credit_data: CreditData):
    # Convert the input data into a pandas DataFrame
    # expected_cols = ['Income', 'Limit', 'Cards', 'Age', 'Education', 'Gender', 'Student', 'Married', 'Ethnicity', 'Balance']
    selected_features = ['Limit', 'Income', 'Age', 'Balance', 'Education']
    input_data = pd.DataFrame([credit_data.dict()])
    input_data = input_data[selected_features]
    print(input_data)
    # Get prediction from the model
    prediction = model.predict(input_data)
    
    return {"prediction": prediction.tolist()}




