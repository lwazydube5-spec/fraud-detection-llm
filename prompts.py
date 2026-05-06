"""
Prompt templates for fraud investigation briefs
===========================================
Contains:
  - build_investigation_prompt()  builds the prompt from claim + model output
  - parse_brief()                 parses LLM text into structured sections

Design principles:
  - Prompt is explicit and structured — numbered sections force consistent output
  - Each section is clearly labelled so parse_brief() can split reliably
  - Instructions are specific — 5 questions not 'some questions'
  - Role context is given — 'experienced fraud investigator' improves output quality

Usage:
    from prompts import build_investigation_prompt, parse_brief
"""


def build_investigation_prompt(claim: dict, prediction: dict, shap_reasons: list) -> str:
    """
    Build a structured investigation prompt from claim data and model output.

    Args:
        claim        — raw claim fields from the API request
        prediction   — model output including fraud_probability and risk_tier
        shap_reasons — top 5 SHAP features from /explain endpoint

    Returns:
        formatted prompt string ready to send to Claude
    """
    # Format SHAP reasons into readable lines
    reasons_text = '\n'.join([
        f"  - {r['feature']}: {r['direction'].replace('_', ' ')} "
        f"(impact: {r['impact']:.4f})"
        for r in shap_reasons
    ])

    return f"""You are an experienced insurance fraud investigator with 15 years of experience.
A machine learning model has flagged the following claim for investigation.
Your job is to analyse the evidence and provide actionable guidance.

CLAIM DETAILS:
- Claimant age         : {claim.get('Age', 'Unknown')}
- Vehicle make         : {claim.get('Make', 'Unknown')}
- Accident area        : {claim.get('AccidentArea', 'Unknown')}
- Fault                : {claim.get('Fault', 'Unknown')}
- Policy type          : {claim.get('PolicyType', 'Unknown')}
- Police report filed  : {claim.get('PoliceReportFiled', 'Unknown')}
- Witness present      : {claim.get('WitnessPresent', 'Unknown')}
- Past claims          : {claim.get('PastNumberOfClaims', 'Unknown')}
- Address change       : {claim.get('AddressChange_Claim', 'Unknown')}
- Days policy to accident : {claim.get('Days_Policy_Accident', 'Unknown')}
- Days policy to claim    : {claim.get('Days_Policy_Claim', 'Unknown')}
- Agent type           : {claim.get('AgentType', 'Unknown')}
- Number of supplements: {claim.get('NumberOfSuppliments', 'Unknown')}

MODEL ASSESSMENT:
- Fraud probability : {prediction.get('fraud_probability', 0):.1%}
- Risk tier         : {prediction.get('risk_tier', 'Unknown')}
- Confidence        : {prediction.get('confidence', 'Unknown')}

TOP FRAUD INDICATORS FROM SHAP ANALYSIS:
{reasons_text}

Based on this evidence provide the following — use exactly these headers:

1. SUMMARY
Write 2 to 3 sentences explaining in plain English why this claim was flagged
and what the overall risk picture looks like.

2. RED FLAGS
List 3 to 5 specific suspicious elements as bullet points.
Be specific — reference actual values from the claim.

3. INVESTIGATION QUESTIONS
List exactly 5 targeted questions to ask the claimant during the investigation call.
Each question should probe a specific suspicious element.

4. VERIFICATION CHECKLIST
List specific documents and evidence to request from the claimant as bullet points.

5. RECOMMENDED ACTION
State one of: APPROVE / INVESTIGATE / ESCALATE / DENY
Follow with exactly one sentence justifying the recommendation.

Be concise, specific, and actionable. Reference actual claim values not generalities."""


def parse_brief(text: str) -> dict:
    """
    Parse LLM investigation brief text into structured sections.

    Args:
        text — raw text response from Claude

    Returns:
        dictionary with keys: summary, red_flags, investigation_questions,
        verification_checklist, recommended_action
    """
    sections = {
        'summary'                : '',
        'red_flags'              : [],
        'investigation_questions': [],
        'verification_checklist' : [],
        'recommended_action'     : '',
    }

    current_section = None

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Detect section headers
        line_upper = line.upper()
        if 'SUMMARY' in line_upper and any(c in line for c in ['#', '1.']):
            current_section = 'summary'
        elif 'RED FLAG' in line_upper and any(c in line for c in ['#', '2.']):
            current_section = 'red_flags'
        elif 'INVESTIGATION' in line_upper and any(c in line for c in ['#', '3.']):
            current_section = 'investigation_questions'
        elif 'VERIFICATION' in line_upper and any(c in line for c in ['#', '4.']):
            current_section = 'verification_checklist'
        elif 'RECOMMENDED' in line_upper and any(c in line for c in ['#', '5.']):
            current_section = 'recommended_action'
        elif current_section:
            # Add content to current section
            if isinstance(sections[current_section], list):
                # Strip bullet point markers
                clean = line.replace('**', '').lstrip('-•*123456789. ').strip()
                if clean:
                    sections[current_section].append(clean)
            else:
                sections[current_section] += line + ' '

    # Clean up strings
    sections['summary'] = sections['summary'].strip()
    sections['recommended_action'] = sections['recommended_action'].replace('**', '').strip()

    return sections


if __name__ == '__main__':
    # Test the prompt and parser with a sample claim
    from bedrock import call_claude

    sample_claim = {
        'Age'                  : 28,
        'Make'                 : 'Honda',
        'AccidentArea'         : 'Urban',
        'Fault'                : 'Policy Holder',
        'PolicyType'           : 'Sedan - Collision',
        'PoliceReportFiled'    : 'No',
        'WitnessPresent'       : 'No',
        'PastNumberOfClaims'   : '2 to 4',
        'AddressChange_Claim'  : 'under 6 months',
        'Days_Policy_Accident' : '1 to 7',
        'Days_Policy_Claim'    : '8 to 15',
        'AgentType'            : 'External',
        'NumberOfSuppliments'  : '3 to 5',
    }

    sample_prediction = {
        'fraud_probability': 0.4673,
        'risk_tier'        : 'HIGH',
        'confidence'       : 'MEDIUM',
    }

    sample_shap = [
        {'feature': 'Fault',              'impact': 0.0465, 'direction': 'increases_fraud'},
        {'feature': 'BasePolicy_Liability','impact': 0.0302, 'direction': 'increases_fraud'},
        {'feature': 'PolicyType_Sedan',   'impact': 0.0256, 'direction': 'decreases_fraud'},
        {'feature': 'AgeOfVehicle',       'impact': 0.0194, 'direction': 'decreases_fraud'},
        {'feature': 'AddressChange_Claim','impact': 0.0187, 'direction': 'increases_fraud'},
    ]

    print('Building prompt...')
    prompt = build_investigation_prompt(sample_claim, sample_prediction, sample_shap)

    print('Sending to Claude...')
    raw_response = call_claude(prompt)

    print('\n=== RAW RESPONSE ===')
    print(raw_response)

    print('\n=== PARSED BRIEF ===')
    brief = parse_brief(raw_response)
    for section, content in brief.items():
        print(f'\n{section.upper()}:')
        if isinstance(content, list):
            for item in content:
                print(f'  • {item}')
        else:
            print(f'  {content}')