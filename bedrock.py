"""
Amazon Bedrock API wrapper
=========================================
Wraps all Bedrock API calls in one place.
All other files import from here — never call boto3 directly.

Usage:
    from bedrock import call_claude
    response = call_claude("What is fraud detection?")
"""

import boto3
import json

REGION   = 'us-east-1'
MODEL_ID = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'

def call_claude(prompt: str, max_tokens: int = 1000) -> str:
    """
    Send a prompt to Claude Haiku and return the response text.

    Args:
        prompt     — the message to send
        max_tokens — maximum length of response (default 1000)

    Returns:
        response text as a string
    """
    bedrock  = boto3.client('bedrock-runtime', region_name=REGION)

    response = bedrock.invoke_model(
        modelId = MODEL_ID,
        
        body    = json.dumps({
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens'        : max_tokens,
            'messages'          : [
                {'role': 'user', 'content': prompt}
            ]
        })
    )

    result = json.loads(response['body'].read())
    return result['content'][0]['text']


if __name__ == '__main__':
    # Quick smoke test — run this file directly to confirm Bedrock works
    print('Testing Bedrock connection...')
    response = call_claude('In one sentence what is insurance fraud detection?')
    print(f'Response: {response}')
    print('Bedrock connection confirmed.')