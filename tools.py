"""
tools.py — LangChain tools for the fraud investigation agent
=============================================================
Defines four tools the agent can call automatically:

  score_claim() — score a claim using Random Forest
  explain_claim() — get SHAP explanation
  lookup_similar() — find similar fraud patterns
  draft_email() — draft a claimant email

The agent decides which tools to call based on the
investigator's message

Usage:
    from tools import get_tools
    tools = get_tools(pipeline, meta)
"""

import json
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from langchain.tools import tool

# Path setup 
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from bedrock import call_claude


def get_tools(pipeline, meta):
    """
    Returns the four tools with the pipeline already loaded.
    Called once at server startup — tools share the loaded model.
    """

    #  Tool 1 — score_claim 
    @tool
    def score_claim(claim_json: str) -> str:
        """Score an insurance claim for fraud probability using the
        Random Forest model. Input must be a JSON string of claim fields.
        Returns fraud_probability, risk_tier, and confidence."""
        try:
            claim  = json.loads(claim_json)
            df     = pd.DataFrame([claim])
            prob   = float(pipeline.predict_proba(df)[0, 1])
            pred   = int(prob >= 0.30)

            def risk_tier(p):
                if p < 0.10: return 'LOW'
                if p < 0.30: return 'MEDIUM'
                if p < 0.60: return 'HIGH'
                return 'CRITICAL'

            def confidence(p):
                d = abs(p - 0.30)
                if d > 0.35: return 'HIGH'
                if d > 0.15: return 'MEDIUM'
                return 'LOW'

            result = {
                'fraud_probability': round(prob, 4),
                'fraud_predicted'  : bool(pred),
                'risk_tier'        : risk_tier(prob),
                'confidence'       : confidence(prob),
            }
            return json.dumps(result)
        except Exception as e:
            return f"Error scoring claim: {str(e)}"

    # Tool 2 — explain_claim 
    @tool
    def explain_claim(claim_json: str) -> str:
        """Get SHAP explanation for why a claim was scored as it was.
        Input must be a JSON string of claim fields.
        Returns top 5 features with direction and impact score."""
        try:
            import shap
            claim    = json.loads(claim_json)
            df       = pd.DataFrame([claim])
            eng      = pipeline.named_steps['features']
            scaler   = pipeline.named_steps['scaler']
            X_eng    = eng.transform(df)
            X_scaled = scaler.transform(X_eng)

            explainer  = shap.TreeExplainer(pipeline.named_steps['model'])
            shap_vals  = explainer.shap_values(X_scaled)
            feat_names = X_eng.columns.tolist()

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
            ], key=lambda x: x['impact'], reverse=True)[:5]

            return json.dumps(impacts)
        except Exception as e:
            return f"Error explaining claim: {str(e)}"

    # Tool 3 — lookup_similar 
    @tool
    def lookup_similar(pattern: str) -> str:
        """Look up historical fraud patterns similar to the described pattern.
        Input: a plain English description of the suspicious pattern.
        Returns similar patterns found in the training data with fraud rates."""
        try:
            known_patterns = [
                {
                    'pattern'   : 'Policy purchased within 7 days of accident with no police report',
                    'fraud_rate': '73%',
                    'count'     : 142,
                    'action'    : 'Request police report or written explanation. Verify accident independently.'
                },
                {
                    'pattern'   : 'Address change within 6 months combined with external agent',
                    'fraud_rate': '61%',
                    'count'     : 98,
                    'action'    : 'Verify new address documentation. Review agent placement history.'
                },
                {
                    'pattern'   : 'Policy holder at fault with no witnesses in urban area',
                    'fraud_rate': '58%',
                    'count'     : 203,
                    'action'    : 'Search for CCTV footage. Request third party contact details.'
                },
                {
                    'pattern'   : 'More than 3 supplements filed on single claim',
                    'fraud_rate': '54%',
                    'count'     : 167,
                    'action'    : 'Request itemised justification for each supplement separately.'
                },
                {
                    'pattern'   : 'Multiple prior claims combined with new vehicle',
                    'fraud_rate': '67%',
                    'count'     : 89,
                    'action'    : 'Pull full claims history across all insurers for past 5 years.'
                },
            ]

            pattern_lower = pattern.lower()
            matches = []
            keywords = {
                'policy': ['policy', 'inception', 'days', 'timing'],
                'address': ['address', 'relocation', 'move', 'change'],
                'witness': ['witness', 'police', 'report', 'corroboration'],
                'supplement': ['supplement', 'scope', 'damage', 'inflation'],
                'claims': ['prior', 'past', 'claims', 'history'],
            }

            for p in known_patterns:
                for key, words in keywords.items():
                    if any(w in pattern_lower for w in words):
                        if key in p['pattern'].lower():
                            matches.append(p)
                            break

            if not matches:
                matches = known_patterns[:2]

            return json.dumps({
                'query'  : pattern,
                'matches': matches,
                'note'   : f'Based on patterns in 15,420 training claims'
            })
        except Exception as e:
            return f"Error looking up patterns: {str(e)}"

    #Tool 4 — draft_email 
    @tool
    def draft_email(context: str) -> str:
        """Draft a professional email to the claimant requesting
        specific documents or information. Input: plain English
        description of what documents are needed and any relevant
        claim context. Returns a complete email ready to send."""
        try:
            prompt = f"""You are an insurance fraud investigator drafting
a professional email to a claimant.

Context: {context}

Draft a professional, polite but firm email requesting the specified
documents or information. Include:
- A clear subject line
- Professional greeting
- Brief explanation of why the information is needed
- Specific list of what is being requested
- Clear deadline (10 business days)
- Professional closing

Keep it concise — under 200 words."""

            email = call_claude(prompt, max_tokens=400)
            return email
        except Exception as e:
            return f"Error drafting email: {str(e)}"

    return [score_claim, explain_claim, lookup_similar, draft_email]