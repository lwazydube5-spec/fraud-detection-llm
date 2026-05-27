"""
langchain_agent.py — LangChain conversational fraud investigation agent
========================================================================
Builds a tool-calling agent with conversation memory using
Amazon Bedrock Claude Haiku 4.5.

The agent can:
  - Score claims using the Random Forest model
  - Explain predictions using SHAP
  - Look up similar historical fraud patterns
  - Draft professional emails to claimants
  - Remember the full conversation history

Usage:
    from langchain_agent import build_agent
    executor = build_agent(pipeline, meta)
    response = executor.invoke({'input': 'Score this claim...'})
"""

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from tools import get_tools


def build_agent(pipeline, meta):
    """
    Build and return the LangGraph react agent.
    Called once at server startup.
    """

    # LLM — Claude Haiku 4.5 via Bedrock 
    llm = ChatBedrock(
        model_id     = 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
        region_name  = 'us-east-1',
        model_kwargs = {'max_tokens': 2000},

    )

    # Tools 
    tools = get_tools(pipeline, meta)

    # System prompt 
    system_prompt = """You are an expert insurance fraud investigator
with access to a machine learning fraud detection system.

You have four tools available:
- score_claim: score a claim for fraud probability
- explain_claim: get SHAP explanation for a prediction
- lookup_similar: find similar historical fraud patterns
- draft_email: draft a professional email to the claimant

When an investigator asks about a claim always use the tools
to get real data — never guess or make up fraud probabilities.

Be concise and actionable. Focus on what the investigator
needs to do next. Reference actual claim values in your responses.

IMPORTANT: When asked to draft an email always call the draft_email 
tool and show the complete email text in your response."""

    # ── Agent ─────────────────────────────────────────────────
    agent = create_agent(
        model  = llm,
        tools  = tools,
        system_prompt = system_prompt,
    )

    return agent


if __name__ == '__main__':
    import sys
    import joblib
    import json
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    MODEL_PATH = Path(__file__).parent / 'models' / 'fraud_model.pkl'
    META_PATH  = Path(__file__).parent / 'models' / 'model_meta.json'

    print('Loading model...')
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)

    print('Building agent...')
    executor = build_agent(pipeline, meta)

    print('Testing agent...')
    agent = build_agent(pipeline, meta)
    response = agent.invoke({
        'messages': [HumanMessage(content='What tools do you have available?')]
    })
    print(f'\nAgent: {response["messages"][-1].content}')
    print('\nLangChain agent confirmed working.')