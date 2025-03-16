# verify_data.py
import os
import weaviate
from dotenv import load_dotenv
from db import generate_embedding
# Load API keys from .env
load_dotenv()

OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Connect to Weaviate Cloud Instance
client = weaviate.Client(
    url="https://wqcicpf8s9coaxew9agmna.c0.europe-west3.gcp.weaviate.cloud",  # Replace with your cloud endpoint
    additional_headers={
        "X-OpenAI-Api-Key": OPENAI_TOKEN,
        "Authorization": f"Bearer {WEAVIATE_API_KEY}"
    }
)

# Fetch all records from ChatHistory
def find_most_similar_entry(query_text):
    """Find the most relevant assistant response based on user query."""
    query_embedding = generate_embedding(query_text)

    # Check if embeddings exist for the stored data
    response = client.query.get("ChatHistory", ["role", "content", "embedding"]).do()
    print("Embedding Check:")
    embedding_exists = False
    for record in response["data"]["Get"]["ChatHistory"]:
        if record.get("embedding") is None or len(record["embedding"]) == 0:
            print("❌ Missing or empty embedding for:", record["content"][:100])
        else:
            print("✅ Embedding exists for:", record["content"][:100])
            embedding_exists = True

    if not embedding_exists:
        print("❌ No valid embeddings found in ChatHistory.")
        return None

    # Check if the schema has the correct vectorizer configuration
    print("\nCurrent Schema:")
    schema = client.schema.get()
    print(schema)
    chat_history_class = next((cls for cls in schema["classes"] if cls["class"] == "ChatHistory"), None)
    if not chat_history_class or chat_history_class.get("vectorizer") != "text2vec-openai":
        print("❌ Schema is missing or vectorizer is incorrect.")
        return None

    # Perform the original search to find the most similar user query
    print("\nPerforming Role Filtered Search for User Query:")
    response = client.query.get(
        "ChatHistory", ["role", "content", "_additional { id }"]
    ).with_near_vector({
        "vector": query_embedding,
        "certainty": 0.75
    }).with_where({
        "operator": "Equal",
        "path": ["role"],
        "valueText": "user"
    }).with_limit(1).do()

    print("Filtered Search Response:", response)

    # Check if response has results
    if "data" in response and "Get" in response["data"] and "ChatHistory" in response["data"]["Get"]:
        if not response["data"]["Get"]["ChatHistory"]:
            print("❌ No similar user entries found.")
            return None

        user_message = response["data"]["Get"]["ChatHistory"][0]
        user_message_id = user_message["_additional"]["id"]

        # Use the same query embedding to find the most relevant assistant response
        print("\nPerforming Vector Search for Most Relevant Assistant Response:")
        response = client.query.get(
            "ChatHistory", ["role", "content"]
        ).with_near_vector({
            "vector": query_embedding,
            "certainty": 0.7  # Adjust if needed
        }).with_where({
            "operator": "Equal",
            "path": ["role"],
            "valueText": "assistant"
        }).with_limit(1).do()

        print("Most Relevant Assistant Response:", response)

        # Return the most relevant assistant's response if found
        if "data" in response and "Get" in response["data"] and "ChatHistory" in response["data"]["Get"]:
            if response["data"]["Get"]["ChatHistory"]:
                return response["data"]["Get"]["ChatHistory"][0]["content"]
            else:
                print("❌ No assistant response found.")
                return None

    print("❌ No similar entries found.")
    return None  # No similar user query or assistant response found

#print(find_most_similar_entry("Have you signed the agreement & paid the invoice?"))

