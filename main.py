from fastapi import FastAPI
import pandas as pd
import xgboost as xgb

app = FastAPI()

model = xgb.XGBClassifier()
model.load_model("model.json")

@app.post("/predict")
def predict(match: dict):
    df = pd.DataFrame([match])
    p = model.predict_proba(df)[0]

    return {
        "home": float(p[1]),
        "draw": float(p[0]),
        "away": float(p[2])
    }
