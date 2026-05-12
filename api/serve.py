"""
api/serve.py — Fraud Detection LLM Investigation API
=====================================================
FastAPI server combining Random Forest fraud scoring,
SHAP explainability, and Amazon Bedrock LLM to generate
structured investigation briefs.

Endpoints:
  POST /investigate  — score + SHAP + LLM investigation brief
  POST /predict      — score only (no LLM)
  GET  /health       — health check
  GET  /docs         — interactive Swagger UI

Usage:
    uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_agent import build_agent
from langchain_core.messages import AIMessage

# Path setup —————————————————————————————————————————————
# Add parent directories to path so we can import bedrock and prompts
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add fraud_det/src to path so joblib can find FraudFeatureEngineer
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bedrock import call_claude
from prompts import build_investigation_prompt, parse_brief

# Config —————————————————————————————————————————————————————
# Path to the trained fraud detection model from the original project
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'fraud_model.pkl'
META_PATH  = Path(__file__).parent.parent / 'models' / 'model_meta.json'

THRESHOLD  = 0.30

# Load model ———————————————————————————
print(f'Loading model from {MODEL_PATH}')
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f'Model not found at {MODEL_PATH}\n'
        f'Run python src/train.py in the fraud_det project first.'
    )

pipeline = joblib.load(MODEL_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

print(f'Model loaded — ROC-AUC: {meta.get("cv_roc_auc", "unknown")}')

print('Building LangChain agent...')
agent = build_agent(pipeline, meta)
print('Agent ready.')

# App ————
app = FastAPI(
    title       = 'Fraud Detection LLM Investigation API',
    description = (
        'Combines Random Forest fraud scoring, SHAP explainability, '
        'and Amazon Bedrock Claude Haiku to generate structured '
        'investigation briefs for insurance fraud analysts.'
    ),
    version = '1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Schema —————
class ClaimInput(BaseModel):
    Month              : str
    WeekOfMonth        : int   = Field(..., ge=1, le=5)
    DayOfWeek          : str
    Make               : str
    AccidentArea       : str
    DayOfWeekClaimed   : str
    MonthClaimed       : str
    WeekOfMonthClaimed : int   = Field(..., ge=1, le=5)
    Sex                : str
    MaritalStatus      : str
    Age                : int   = Field(..., ge=0, le=120)
    Fault              : str
    PolicyType         : str
    VehicleCategory    : str
    VehiclePrice       : str
    PolicyNumber       : int
    RepNumber          : int
    Deductible         : int
    DriverRating       : int   = Field(..., ge=1, le=4)
    Days_Policy_Accident : str
    Days_Policy_Claim  : str
    PastNumberOfClaims : str
    AgeOfVehicle       : str
    AgeOfPolicyHolder  : str
    PoliceReportFiled  : str
    WitnessPresent     : str
    AgentType          : str
    NumberOfSuppliments: str
    AddressChange_Claim: str
    NumberOfCars       : str
    Year               : int
    BasePolicy         : str

    class Config:
        json_schema_extra = {
            'example': {
                'Month': 'Jan', 'WeekOfMonth': 3, 'DayOfWeek': 'Monday',
                'Make': 'Honda', 'AccidentArea': 'Urban',
                'DayOfWeekClaimed': 'Wednesday', 'MonthClaimed': 'Feb',
                'WeekOfMonthClaimed': 2, 'Sex': 'Male',
                'MaritalStatus': 'Single', 'Age': 28,
                'Fault': 'Policy Holder', 'PolicyType': 'Sedan - Collision',
                'VehicleCategory': 'Sedan', 'VehiclePrice': 'more than 69000',
                'PolicyNumber': 99999, 'RepNumber': 5, 'Deductible': 400,
                'DriverRating': 1, 'Days_Policy_Accident': '1 to 7',
                'Days_Policy_Claim': '8 to 15', 'PastNumberOfClaims': '2 to 4',
                'AgeOfVehicle': 'new', 'AgeOfPolicyHolder': '26 to 30',
                'PoliceReportFiled': 'No', 'WitnessPresent': 'No',
                'AgentType': 'External', 'NumberOfSuppliments': '3 to 5',
                'AddressChange_Claim': 'under 6 months',
                'NumberOfCars': '1 vehicle', 'Year': 1994,
                'BasePolicy': 'Collision'
            }
        }

# Helpers ————————————————————————————————
def get_risk_tier(prob: float) -> str:
    if prob < 0.10: return 'LOW'
    if prob < 0.30: return 'MEDIUM'
    if prob < 0.60: return 'HIGH'
    return 'CRITICAL'

def get_confidence(prob: float) -> str:
    distance = abs(prob - THRESHOLD)
    if prob < 0.30:
        if distance > 0.20: return 'HIGH'
        if distance > 0.10: return 'MEDIUM'
        return 'LOW'
    else:
        if distance > 0.35: return 'HIGH'
        if distance > 0.10: return 'MEDIUM'
        return 'LOW'

def score_claim(claim_data: dict) -> dict:
    """Score a single claim using the Random Forest model."""
    df   = pd.DataFrame([claim_data])
    prob = float(pipeline.predict_proba(df)[0, 1])
    pred = int(prob >= THRESHOLD)
    return {
        'fraud_probability': round(prob, 4),
        'fraud_predicted'  : bool(pred),
        'risk_tier'        : get_risk_tier(prob),
        'confidence'       : get_confidence(prob),
    }

def get_shap_reasons(claim_data: dict) -> list:
    """Get top 5 SHAP reasons for a prediction."""
    import shap
    df       = pd.DataFrame([claim_data])
    eng      = pipeline.named_steps['features']
    scaler   = pipeline.named_steps['scaler']
    X_eng    = eng.transform(df)
    X_scaled = scaler.transform(X_eng)

    explainer   = shap.TreeExplainer(pipeline.named_steps['model'])
    shap_vals   = explainer.shap_values(X_scaled)
    feat_names  = X_eng.columns.tolist()

    if isinstance(shap_vals, list):
        fraud_shaps = shap_vals[1][0]
    elif shap_vals.ndim == 3:
        fraud_shaps = shap_vals[0, :, 1]
    else:
        fraud_shaps = shap_vals[0]

    impacts = sorted([
        {
            'feature'  : feat_names[i],
            'impact'   : round(abs(float(fraud_shaps[i])), 4),
            'direction': 'increases_fraud' if fraud_shaps[i] > 0 else 'decreases_fraud',
        }
        for i in range(len(feat_names))
    ], key=lambda x: x['impact'], reverse=True)

    return impacts[:5]

# Endpoints –––––––––––––––––––––

@app.post('/predict')
def predict(claim: ClaimInput):
    """Score a single claim — Random Forest only, no LLM."""
    try:
        result = score_claim(claim.model_dump())
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/investigate')
def investigate(claim: ClaimInput):
    """
    Full investigation brief — Random Forest + SHAP + Bedrock LLM.
    Returns fraud score, top SHAP reasons, and a structured
    investigation brief generated by Claude Haiku.
    """
    try:
        t0         = time.time()
        claim_data = claim.model_dump()

        # Step 1 — score the claim
        prediction = score_claim(claim_data)

        # Step 2 — get SHAP reasons
        shap_reasons = get_shap_reasons(claim_data)

        # Step 3 — build prompt and call Bedrock
        prompt     = build_investigation_prompt(claim_data, prediction, shap_reasons)
        raw_brief  = call_claude(prompt)
        brief      = parse_brief(raw_brief)

        return JSONResponse(content={
            'fraud_probability'  : prediction['fraud_probability'],
            'fraud_predicted'    : prediction['fraud_predicted'],
            'risk_tier'          : prediction['risk_tier'],
            'confidence'         : prediction['confidence'],
            'top_reasons'        : shap_reasons,
            'investigation_brief': brief,
            'total_ms'           : round((time.time() - t0) * 1000, 2),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message : str
    claim   : Optional[ClaimInput] = None
    history : list = []


@app.post('/chat')
def chat(request: ChatRequest):
    """
    Conversational fraud investigation agent.
    The agent decides which tools to call based on the message.
    Optionally accepts a claim for context.
    """
    try:
        t0 = time.time()

        # build the message — include claim context if provided
        if request.claim:
            claim_json = json.dumps(request.claim.model_dump())
            full_message = (
                f"{request.message}\n\n"
                f"Claim data: {claim_json}"
            )
        else:
            full_message = request.message

        # invoke the agent
        # build message history
        messages = []
        for turn in request.history:
            if turn.get('role') == 'user':
                messages.append(HumanMessage(content=turn['content']))
            elif turn.get('role') == 'assistant':
                messages.append(AIMessage(content=turn['content']))

        # add current message
        messages.append(HumanMessage(content=full_message))

        # invoke the agent
        response = agent.invoke({'messages': messages})

        # get the last message — the agent's final response
        answer = response['messages'][-1].content

        # build updated history to return to client
        updated_history = request.history + [
            {'role': 'user',      'content': full_message},
            {'role': 'assistant', 'content': answer},
        ]

        return JSONResponse(content={
            'response' : answer,
            'history'  : updated_history,
            'total_ms' : round((time.time() - t0) * 1000, 2),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
def health():
    """Health check — confirms model and Bedrock are ready."""
    return {
        'status'     : 'healthy',
        'model_type' : meta.get('model_type', 'unknown'),
        'threshold'  : THRESHOLD,
        'cv_roc_auc' : meta.get('cv_roc_auc', 'unknown'),
        'bedrock'    : 'claude-haiku-4-5',
    }


@app.get('/')
def root():
    return {
        'name'      : 'Fraud Detection LLM Investigation API',
        'version'   : '1.0.0',
        'endpoints' : ['/predict', '/investigate', '/health', '/docs'],
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)