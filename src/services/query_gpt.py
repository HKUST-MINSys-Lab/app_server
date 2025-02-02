import openai
import os
from src.utils.err import UnknownModelErr

def query(prompt, model, retry=3):
    if model.startswith('gpt'):
        api_url = os.getenv('OPENAI_API_URL')
        api_key = os.getenv('OPENAI_API_KEY')
    elif model == 'deepseek-chat' or model == 'deepseek-reasoner':
        api_url = os.getenv('DEEPSEEK_API_URL')
        api_key = os.getenv('DEEPSEEK_API_KEY')
    # elif local llm server
    # elif claude special client
    else:
        raise UnknownModelErr
    client = openai.OpenAI(
        base_url=api_url,
        api_key=api_key
    )
    while retry > 0:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            retry -= 1
            if retry == 0:
                raise e

    return None