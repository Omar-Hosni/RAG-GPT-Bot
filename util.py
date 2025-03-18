import os
import random
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np
import json
import time
import asyncio
from transformers import pipeline
from pysentimiento import create_analyzer

emotion_analyzer = create_analyzer(task="emotion", lang="en")

greetings = [
    "hi", "hey", "hello", "yo", "sup", "what's up", "howdy", "hiya",
    "hey there", "hello there", "how's it going", "how's everything",
    "what's happening", "good day", "greetings", "salutations",
    "hi there", "how are you", "how have you been", "how do you do",
    "what's good", "what's new", "morning", "good morning", "good afternoon",
    "good evening", "evening", "hiya there", "aloha", "hola", "bonjour",
    "hallo", "ciao", "namaste", "salaam", "shalom", "konnichiwa", "annyeong"
]

CONVERSATION_FILE = "conversation.json"
EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.50   # Tune this based on testing
openai.api_key = os.getenv("OPENAI_TOKEN")


def read_conversation_file():
    try:
        with open(CONVERSATION_FILE, "r", encoding="utf-8") as file:
            conversation_history = json.load(file)
    except FileNotFoundError:
        conversation_history = []
    return conversation_history

def update_conversation_history(curr_conversation):
    with open(CONVERSATION_FILE, "w", encoding="utf-8") as file:
        file.write(json.dumps(curr_conversation, ensure_ascii=False, indent=4).encode('utf-8'))
    print("Conversation history updated successfully.")

def is_greeting(message: str) -> bool:
    return message.lower() in greetings

def generate_embedding(query, engine):
    return get_embedding(query, engine)

def find_cos_similarity(current_embedding, stored_embedding):
    return cosine_similarity(current_embedding, stored_embedding)

def detect_emotion(user, msg):
    result = emotion_analyzer.predict(str(msg))
    emotion = None
    for label in result.probas:
        if label in ["joy", "anger", "sadness", "fear"] and result.probas[label] > 0.85:
            emotion = f"User {user} said {msg} feeling emotion: {label}, with percentage: {result.probas[label] * 100:.2f}%"

    return emotion

async def send_heartbeats():
    """Send manual heartbeats to Discord to prevent disconnections."""
    while True:
        await asyncio.sleep(5)  # Send heartbeat every 5 seconds
        try:
            await client.ws.ping()
            print("💓 Sent manual heartbeat to Discord.")
        except Exception as e:
            print(f"❌ Failed to send heartbeat: {e}")
            break

def is_business_or_social_media_related(message: str) -> bool:
    """Check if a message is related to business or social media using pysentimiento."""
    return False

def is_travel_related(message: str) -> bool:

    if len(message) < 3:
        return False

    try:
        temp_conversation = [
            {"role": "system", "content": "You are a travel expert. Your job is to identify if a given question is related to travel, such as trips, locations, or travel plans. If it is unrelated, say 'no'. If it is travel-related, say 'yes'."},
            {"role": "user", "content": f"Is this message about traveling? {message}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=temp_conversation,
            max_tokens=50
        )
        return "yes" in response['choices'][0]['message']['content'].strip().lower()
    except Exception as e:
        print(f"Error in travel detection: {e}")
        return False


def is_worth_learning(message: str) -> bool:
    """Uses GPT to determine if a business-related message is valuable for learning."""
    
    try:
        prompt = f"""
        A user has asked the following business-related question:
        "{message}"
        
        Should this question be stored for future retrieval in a knowledge base?
        If it's a common, useful, and informative question that would help a business ai bot develop more awareness, say "yes".
        If it's too vague, unhelpful, or meaningless, say "no".
        """

        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "system", "content": "You are a business assistant that evaluates if a question is worth storing."},
                      {"role": "user", "content": prompt}],
            max_tokens=10
        )

        return "yes" in response['choices'][0]['message']['content'].strip().lower()

    except Exception as e:
        print(f"⚠️ Error in learning evaluation: {e}")
        return False  # If error, don't store



def transcribe_video_and_learn_it(youtube_url):
    pass


def has_majority_common_words(str1: str, str2: str, threshold: float = 60.0) -> bool:
    # Convert strings to sets of words
    words1 = set(str1.split())
    words2 = set(str2.split())

    # Find common words
    common_words = words1 & words2

    # Calculate the percentage of overlap (relative to the smaller set)
    smaller_set_size = min(len(words1), len(words2))
    if smaller_set_size == 0:  # Avoid division by zero
        return False

    overlap_percentage = (len(common_words) / smaller_set_size) * 100

    # Check if the overlap exceeds the threshold
    return overlap_percentage > threshold


def introduce_typos(text, typo_percentage=0.02):
    words = text.split()
    num_typos = max(1, int(len(words) * typo_percentage))  # At least one typo if the text is very short
    
    typo_indices = random.sample(range(len(words)), num_typos)
    
    for index in typo_indices:
        word = words[index]
        if len(word) > 1:
            typo_type = random.choice(['swap', 'remove'])
            if typo_type == 'swap':
                # Swap two adjacent letters
                pos = random.randint(0, len(word) - 2)
                word = list(word)
                word[pos], word[pos + 1] = word[pos + 1], word[pos]
                words[index] = ''.join(word)
            elif typo_type == 'remove':
                # Remove a random letter
                pos = random.randint(0, len(word) - 1)
                words[index] = word[:pos] + word[pos + 1:]
    
    return ' '.join(words)

def humanize_text(text):
    prompt = f"""
        {text}
    humanize this and make it sound normal-like and very short, no need to use numerical or bullet points
    """
    try:
        response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )

        return response['choices'][0]['message']['content'].strip().lower()
    except Exception as e:
        print(f"⚠️ Error in learning evaluation: {e}")
        return False  
