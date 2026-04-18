from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import httpx

app = FastAPI(title="Credit Card Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

df_original = pd.read_csv('df_original.csv')
X_test      = pd.read_csv('X_test.csv', index_col=0)

# Mappings
rMap = {
    'txn':  {'Online Shopping':0.8,'International':1.0,'POS / Swipe':0.3,'ATM Withdrawal':0.5},
    'loc':  {'Low Risk':0.1,'Medium Risk':0.5,'High Risk':0.9},
    'dev':  {'Mobile':0.6,'Desktop':0.4,'ATM Machine':0.5,'POS Terminal':0.3},
    'merch':{'Grocery':0.1,'Electronics':0.5,'Travel':0.6,'Fuel':0.4,'Luxury':0.7,'Gaming/Crypto':1.0},
    'ctype':{'Debit Card':0.3,'Credit Card':0.5,'Prepaid Card':0.8},
    'dist': {'Same City':0.1,'Different City':0.5,'Different Country':0.9},
    'freq': {'1-2 times':0.2,'3-5 times':0.6,'5+ times':1.0},
    'ltxn': {'< 5 minutes':1.0,'5-30 minutes':0.6,'30 min - 2 hours':0.3,'2+ hours':0.1},
    'wknd': {'Yes':0.6,'No':0.3},
    'cage': {'New Card (< 3 months)':0.8,'Regular (3-12 months)':0.4,'Old Card (1+ year)':0.2}
}

class TransactionInput(BaseModel):
    amount: float
    hour: float
    transaction_type: str
    location_risk: str
    device_type: str
    merchant_category: str
    card_type: str
    distance: str
    frequency_today: str
    last_txn_time: str
    is_weekend: str
    card_age: str

# def find_closest_transaction(data: TransactionInput):
#     amount = data.amount
#     hour   = data.hour

#     amt_tol  = amount * 0.2
#     filtered = df_original[
#         (df_original['Amount'] >= amount - amt_tol) &
#         (df_original['Amount'] <= amount + amt_tol)
#     ]
#     if len(filtered) == 0:
#         filtered = df_original

#     filtered2 = filtered[
#         (filtered['Hour'] >= hour - 2) &
#         (filtered['Hour'] <= hour + 2)
#     ]
#     if len(filtered2) == 0:
#         filtered2 = filtered

#     closest      = filtered2.iloc[0]
#     feature_cols = X_test.columns.tolist()
#     features     = []

#     for col in feature_cols:
#         if col in closest.index:
#             features.append(closest[col])
#         elif col == 'Amount_Scaled':
#             features.append((amount - df_original['Amount'].mean()) / df_original['Amount'].std())
#         elif col == 'Time_Scaled':
#             features.append(((hour*3600) - df_original['Time'].mean()) / df_original['Time'].std())
#         else:
#             features.append(0)

#     return features

def find_closest_transaction(data: TransactionInput):
    amount = data.amount
    hour   = data.hour

    # Risk score calculate karo
    risk = (
        rMap['txn'][data.transaction_type] +
        rMap['loc'][data.location_risk] +
        rMap['dev'][data.device_type] +
        rMap['merch'][data.merchant_category] +
        rMap['ctype'][data.card_type] +
        rMap['dist'][data.distance] +
        rMap['freq'][data.frequency_today] +
        rMap['ltxn'][data.last_txn_time] +
        rMap['wknd'][data.is_weekend] +
        rMap['cage'][data.card_age]
    ) / 10

    # Step 1 — Amount filter (20% tolerance)
    amt_tol  = amount * 0.3
    filtered = df_original[
        (df_original['Amount'] >= amount - amt_tol) &
        (df_original['Amount'] <= amount + amt_tol)
    ]
    if len(filtered) == 0:
        filtered = df_original

    # Step 2 — Hour filter (3 hour range)
    filtered2 = filtered[
        (filtered['Hour'] >= hour - 3) &
        (filtered['Hour'] <= hour + 3)
    ]
    if len(filtered2) == 0:
        filtered2 = filtered

    # Step 3 — High risk inputs pe fraud transactions prefer karo
    if risk >= 0.6:
        fraud_rows = filtered2[filtered2['Class'] == 1]
        if len(fraud_rows) > 0:
            filtered2 = fraud_rows

    # Step 4 — Closest row lo
    closest      = filtered2.iloc[0]
    feature_cols = X_test.columns.tolist()
    features     = []

    for col in feature_cols:
        if col in closest.index:
            features.append(closest[col])
        elif col == 'Amount_Scaled':
            features.append((amount - df_original['Amount'].mean()) /
                             df_original['Amount'].std())
        elif col == 'Time_Scaled':
            features.append(((hour*3600) - df_original['Time'].mean()) /
                             df_original['Time'].std())
        else:
            features.append(0)

    return features


@app.get("/")
def home():
    return {"message": "🏦 FraudGuard API is running!"}

@app.post("/predict")
def predict(data: TransactionInput):
    features       = find_closest_transaction(data)
    features_array = np.array(features).reshape(1, -1)

    prediction  = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0][1]

    risk_score = (
        rMap['txn'][data.transaction_type] +
        rMap['loc'][data.location_risk] +
        rMap['dev'][data.device_type] +
        rMap['merch'][data.merchant_category] +
        rMap['ctype'][data.card_type] +
        rMap['dist'][data.distance] +
        rMap['freq'][data.frequency_today] +
        rMap['ltxn'][data.last_txn_time] +
        rMap['wknd'][data.is_weekend] +
        rMap['cage'][data.card_age]
    ) / 10

    # return {
    #     "prediction" : int(prediction),
    #     "verdict"    : "FRAUD" if prediction == 1 else "LEGIT",
    #     "probability": round(float(probability) * 100, 2),
    #     "risk_score" : round(risk_score, 2)
    # }

    ml_prob      = float(probability)
    combined     = (ml_prob * 0.6) + (risk_score * 0.4)
    final_verdict = "FRAUD" if combined > 0.4 else "LEGIT"
    final_prob    = round(combined * 100, 2)

    return {
        "prediction" : int(prediction),
        "verdict"    : final_verdict,
        "probability": final_prob,
        "risk_score" : round(risk_score, 2)
    }

class ExplainInput(BaseModel):
    amount: float
    hour: float
    transaction_type: str
    location_risk: str
    device_type: str
    merchant_category: str
    card_type: str
    distance: str
    frequency_today: str
    last_txn_time: str
    is_weekend: str
    card_age: str
    verdict: str
    probability: float
    risk_score: float

@app.post("/explain")
async def explain(data: ExplainInput):
    prompt = f"""You are a fraud detection expert. Analyze this transaction.

Transaction:
- Amount: Rs {data.amount}
- Hour: {data.hour}:00
- Type: {data.transaction_type}
- Location Risk: {data.location_risk}
- Device: {data.device_type}
- Merchant: {data.merchant_category}
- Card Type: {data.card_type}
- Distance: {data.distance}
- Frequency Today: {data.frequency_today}
- Last Transaction: {data.last_txn_time} ago
- Weekend: {data.is_weekend}
- Card Age: {data.card_age}

ML Result: {data.verdict} ({data.probability}% probability, risk: {data.risk_score})

Return ONLY a JSON object with exactly these 3 keys, no extra text:
{{
  "signal_explanation": "2-3 sentences why this is {data.verdict}",
  "pattern_analysis": "2-3 sentences about time and amount patterns",
  "precautions": ["action 1", "action 2", "action 3", "action 4"]
}}"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=60.0
        )
        result = response.json()
        text = result.get('response', '')

        import json
        try:
            start = text.find('{')
            end   = text.rfind('}') + 1
            return json.loads(text[start:end])
        except:
            return {
                "signal_explanation": text[:300] if text else "Analysis unavailable.",
                "pattern_analysis"  : "Pattern analysis based on transaction data.",
                "precautions"       : ["Monitor your account", "Enable alerts", "Contact bank if needed", "Check recent transactions"]
            }