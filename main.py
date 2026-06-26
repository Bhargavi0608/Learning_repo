from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Optional , List , Union
from enum import Enum

app = FastAPI(title="Loan Approval API", 
    description="API to predict loan approval using a Random Forest model"
)

# Load your model resources
model = joblib.load("random_forest_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# 1. Enums for input validation dropdowns
class HomeOwnership(str, Enum):
    RENT = "RENT"
    MORTGAGE = "MORTGAGE"
    OWN = "OWN"
    OTHER = "OTHER"

class LoanIntent(str, Enum):
    PERSONAL = "PERSONAL"
    EDUCATION = "EDUCATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"

class LoanGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"

class DefaultOnFile(str, Enum):
    Y = "Y"
    N = "N"

# 2. Pydantic request structure
class LoanRequest(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: HomeOwnership       
    person_emp_length: Optional[float] = None 
    loan_intent: LoanIntent                     
    loan_grade: LoanGrade                       
    loan_amnt: float
    loan_int_rate: Optional[float] = None     
    loan_percent_income: float
    cb_person_default_on_file: DefaultOnFile   
    cb_person_cred_hist_length: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 24,
                "person_income": 50000.0,
                "person_home_ownership": "RENT",
                "person_emp_length": 2.0,
                "loan_intent": "EDUCATION",
                "loan_grade": "B",
                "loan_amnt": 10000.0,
                "loan_int_rate": 11.12,
                "loan_percent_income": 0.20,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 3
            }
        }
    }


@app.post("/predict", summary="Predict Loan Approval Status")
def predict_loan(data: Union[LoanRequest, List[LoanRequest]]):
    # 1. Convert incoming JSON data into a dictionary
    input_dict = data.model_dump()
    
    # 2. Convert dictionary to DataFrame FIRST
    df = pd.DataFrame([input_dict])

    # 3. Handle 'id' check safely now that df exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 4. One-hot encode the text data (Changed 'new_data' to 'df')
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # 5. Align columns with your training features template
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ensure identical order and drop extra columns not in training
    df_encoded = df_encoded[model_columns]
        
    # 6. Make the prediction
    prediction = model.predict(df_encoded)[0]
    status = "Approved" if prediction == 1 else "Denied"
    
    return {
        "loan_status": status,
        "prediction_code": int(prediction)
    }
