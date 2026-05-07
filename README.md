# Fraud Detection LLM Investigation System

An end-to-end AI investigation system that combines a trained Random Forest fraud detection model with Amazon Bedrock to generate structured investigation briefs for insurance fraud analysts.

🔗 **[Live Demo](https://lwazydube5-spec.github.io/fraud-investigation-ui)**

---

## What it does

A fraud analyst submits an insurance claim and the system runs three steps automatically:

1. **Random Forest** scores the claim — fraud probability, risk tier, confidence
2. **SHAP** explains which features drove the prediction — top 5 with direction and impact
3. **Amazon Bedrock** (Claude Haiku 4.5) generates a structured investigation brief containing a plain English summary, specific red flags, 5 targeted investigation questions, a document verification checklist, and a recommended action

---

## Architecture

```
Browser form
      ↓
GitHub Pages — static HTML frontend
      ↓
Render — FastAPI Python server
      ↓
Step 1 — Random Forest → fraud_probability, risk_tier, confidence
Step 2 — SHAP TreeExplainer → top 5 feature reasons
Step 3 — Amazon Bedrock → Claude Haiku 4.5 → investigation brief
      ↓
JSON response rendered in browser
```

---

## Project structure

```
fraud_det_llm/
├── api/
│   ├── __init__.py
│   └── serve.py          # FastAPI — /predict, /investigate, /health
├── src/
│   └── features.py       # FraudFeatureEngineer — imported from fraud_det
├── models/
│   ├── fraud_model.pkl   # Trained Random Forest pipeline
│   └── model_meta.json   # Threshold, CV metrics, model type
├── frontend/
│   └── index.html        # Single file dashboard — deployed to GitHub Pages
├── bedrock.py            # Amazon Bedrock wrapper — call_claude()
├── prompts.py            # Prompt engineering — build_investigation_prompt() + parse_brief()
├── requirements.txt
└── Dockerfile
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /predict | Score a claim — Random Forest only, fast |
| POST | /investigate | Score + SHAP + LLM investigation brief |
| GET | /health | System status — model and Bedrock info |
| GET | /docs | Interactive Swagger UI |

---

## Investigation brief output

```json
{
  "fraud_probability": 0.4673,
  "fraud_predicted": true,
  "risk_tier": "HIGH",
  "confidence": "MEDIUM",
  "top_reasons": [
    {"feature": "Fault", "impact": 0.0465, "direction": "increases_fraud"}
  ],
  "investigation_brief": {
    "summary": "...",
    "red_flags": ["...", "..."],
    "investigation_questions": ["...", "..."],
    "verification_checklist": ["...", "..."],
    "recommended_action": "INVESTIGATE ..."
  },
  "total_ms": 18106
}
```

---

## Risk tiers

| Tier | Probability | Action |
|------|-------------|--------|
| LOW | < 10% | Auto-approve |
| MEDIUM | 10% – 30% | Standard review |
| HIGH | 30% – 60% | Priority investigation |
| CRITICAL | > 60% | Immediate escalation |

---

## How the three files connect

| File | Role | Responsibility |
|------|------|----------------|
| `bedrock.py` | Technical layer | HOW to talk to Claude — boto3 Bedrock API call |
| `prompts.py` | Intelligence layer | WHAT to say to Claude — augmented prompt + parser |
| `api/serve.py` | Orchestration layer | WHEN and WHY — runs all three steps, returns response |

The augmented prompt combines 13 human-readable claim fields + model score + SHAP reasons into a structured 500 token prompt. Claude never sees the raw threshold or the 93 engineered features — Python handles the numbers, Claude handles the language.

---



## Tech stack

| Layer | Technology |
|-------|-----------|
| ML model | Random Forest — scikit-learn |
| Explainability | SHAP TreeExplainer |
| LLM | Claude Haiku 4.5 via Amazon Bedrock |
| API | FastAPI + uvicorn |
| Containerisation | Docker |
| Frontend | HTML / CSS / JavaScript |
| API hosting | Render |
| Frontend hosting | GitHub Pages |

---

## Related project

This system is built on top of the base fraud detection model:
🔗 [fraud-detection-ml](https://github.com/lwazydube5-spec/fraud-detection-ml) — Random Forest · SHAP · SageMaker · Model Monitor · CI/CD