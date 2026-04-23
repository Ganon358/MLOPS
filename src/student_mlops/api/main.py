from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(title="Student Performance Prediction API")

model = joblib.load("models/model.joblib")


class StudentInput(BaseModel):
    Gender: str
    Age: int
    District: str
    School_Type: str
    Study_Hours_per_Week: int
    Attendance: int
    Parent_Education: str
    Family_Income_BDT: int
    Internet_Access: str
    Private_Tuition: str
    Previous_GPA: float
    SSC_Result: float


@app.get("/")
def home():
    return {"message": "API ML MLOps fonctionne"}

@app.post("/predict")
def predict(data: StudentInput):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_df)

        return {
            "prediction": float(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}