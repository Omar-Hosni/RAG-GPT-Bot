import openai
import os
from dotenv import load_dotenv
import random
load_dotenv()
from db import retrieve_conversation_history

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_TOKEN")


def generate_nick_response(user_input: str, user_id: str, session_context: list) -> str:

    messages = [{"role": "system", "content": "You are Nik Setting. You are not a bot nor an AI. You are a friendly business strategist and marketing consultant. Your responses should be concise, direct, and results-oriented. You speak like an experienced entrepreneur who focuses on performance metrics, conversions, and growth. Your tone is professional but informal and friendly, similar to a high-level sales consultant advising a client. Keep replies short, actionable, and to the point."}]
    
    for msg in session_context:
        messages.append({
            "role":msg["role"],
            "content":msg["content"]
        })
    messages.append({"role": "user", "content": user_input})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=messages,
            max_tokens=150,
            temperature=random.choice([0.7, 0.8, 0.9]),  # Creative generation
            #top_p=0.95,        # Balanced diversity for natural responses
            #presence_penalty=0.2,  # Discourage repetition of ideas
            #frequency_penalty=0.1  # Avoid repeated words
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"
