import weaviate
import json
import os
import openai
from datetime import datetime
from dotenv import load_dotenv
import requests
import time
import asyncio
#from weaviate.classes.init import Auth

load_dotenv()

OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

openai.api_key = OPENAI_TOKEN

# Connect to Weaviate Local
# client = weaviate.Client(
#     url="http://localhost:8080",  # Use the HTTP endpoint
#     additional_headers={
#         "X-OpenAI-Api-Key": OPENAI_TOKEN  # Pass OpenAI API key for vectorization
#     }
# )

# Connect to Weaviate Online
# client = weaviate.connect_to_weaviate_cloud(
#     cluster_url="https://wqcicpf8s9coaxew9agmna.c0.europe-west3.gcp.weaviate.cloud",
#     auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
# )

# Connect to Weaviate Online
client = weaviate.Client(
    url="https://wqcicpf8s9coaxew9agmna.c0.europe-west3.gcp.weaviate.cloud",  # Replace with your cloud endpoint
    timeout_config=(10, 60),
    additional_headers={
        "X-OpenAI-Api-Key": OPENAI_TOKEN,
        "Authorization": f"Bearer {WEAVIATE_API_KEY}"
    }
)

# Load chat.json file
def load_chat_json(file_path="chat.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Generate embeddings for text
def generate_embedding(text):
    """Generate and return OpenAI embeddings as a list of floats."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    
    embedding = response["data"][0]["embedding"]

    if not isinstance(embedding, list):
        raise ValueError("❌ Embedding is not a list!")
    
    return [float(x) for x in embedding]  # Ensure all elements are floats


def format_rfc3339(dt):
    """Format a datetime object as RFC3339-compliant string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # ✅ Removes microseconds


def store_chat_history():
    chat_data = load_chat_json()

    for entry in chat_data:
        role = entry["role"]
        content = entry["content"]
        timestamp = format_rfc3339(datetime.utcnow())  # ✅ Ensure correct format
        print('current_entry_message:', entry["content"])

        # Generate embedding only for user queries
        embedding = generate_embedding(content) if role == "user" else None  
        try:
            client.data_object.create(
                data_object={
                    "role": role,
                    "content": content,
                    "timestamp": timestamp,  # ✅ Corrected timestamp format
                    "embedding": embedding if embedding else []  
                },
                class_name="ChatHistory"
            )
        except requests.exceptions.RequestException as e:
            print(f"RequestException: {e}")
        except weaviate.exceptions.UnexpectedStatusCodeException as e:
            print(f"Weaviate Exception: {e}")
    
    print("✅ Chat history successfully stored in Weaviate!")


# Ensure schema exists
def create_schema():
    schema = {
        "classes": [
            {
                "class": "ChatHistory",
                "description": "Stores chat history between users and assistant",
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "text2vec-openai": {
                        "model": "text-embedding-3-small",
                        "type": "text"
                    }
                },
                "properties": [
                    {"name": "role", "dataType": ["text"]},
                    {"name": "content", "dataType": ["text"]},
                    {"name": "timestamp", "dataType": ["date"]},
                    {"name": "embedding", "dataType": ["number[]"]}  # ✅ Must be number[]
                ],
            }
        ]
    }

    existing_classes = [c["class"] for c in client.schema.get().get("classes", [])]
    
    if "ChatHistory" not in existing_classes:
        client.schema.create(schema)
        print("✅ ChatHistory schema created in Weaviate.")
    else:
        print("✅ ChatHistory schema already exists.")


# Store a conversation entry in Weaviate
def store_conversation_entry(role, content, message_id):
    """Stores a conversation entry in Weaviate if content is not empty."""
    if not content.strip():
        print("❌ Warning: Empty content, skipping storage.")
        return

    try:
        embedding = generate_embedding(content) if role == "user" else None

        client.data_object.create(
            data_object={
                "role": role,
                "content": content,
                "timestamp": format_rfc3339(datetime.utcnow()),
                "message_id": message_id,
                **({"embedding": embedding} if embedding else {})
            },
            class_name="ChatHistory"  # Ensure class name is consistent
        )
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        print(f"❌ Error storing in Weaviate: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")



# Query Weaviate to Find Similar Messages Local
# def find_most_similar_entry(query_text):
#     """Find the assistant's response for the most similar user query."""
#     query_embedding = generate_embedding(query_text)

#     # Find the most similar user message
#     response = client.query.get(
#         "Conversation", ["role", "content", "_additional { id }"]
#     ).with_near_vector({
#         "vector": query_embedding,
#         "certainty": 0.75  # Adjust similarity threshold
#     }).with_where({
#         "operator": "Equal",
#         "path": ["role"],
#         "valueText": "user"
#     }).with_limit(1).do()

#     print('find most similar entry response: ', response)

#     if response["data"]["Get"]["Conversation"]:
#         user_message = response["data"]["Get"]["Conversation"][0]
#         user_message_id = user_message["_additional"]["id"]

#         # Find the assistant's response that follows this user message
#         response = client.query.get(
#             "Conversation", ["role", "content"]
#         ).with_where({
#             "operator": "And",
#             "operands": [
#                 {
#                     "operator": "Equal",
#                     "path": ["role"],
#                     "valueText": "assistant"
#                 },
#                 {
#                     "operator": "GreaterThan",
#                     "path": ["_additional", "id"],
#                     "valueString": user_message_id
#                 }
#             ]
#         }).with_limit(1).do()

#         # Return the assistant's response if found
#         if response["data"]["Get"]["Conversation"]:
#             return response["data"]["Get"]["Conversation"][0]["content"]

#     return None  # No similar user query or assistant response found


# Query Weaviate to Find Similar Messages Online
async def find_most_similar_entry(query_text):
    """Find the most relevant assistant response based on user query."""
    query_embedding = generate_embedding(query_text)

    # Check if embeddings exist for the stored data
    response = await asyncio.to_thread(client.query.get, "ChatHistory", ["role", "content", "embedding"])
    response = response.do()
    # print("Embedding Check:")
    # embedding_exists = False
    # for record in response["data"]["Get"]["ChatHistory"]:
    #     if record.get("embedding") is None or len(record["embedding"]) == 0:
    #         print("❌ Missing or empty embedding for:", record["content"][:100])
    #     else:
    #         print("✅ Embedding exists for:", record["content"][:100])
    #         embedding_exists = True

    # if not embedding_exists:
    #     print("❌ No valid embeddings found in ChatHistory.")
    #     return None

    # Perform vector search directly to find the most similar user query
    response = await asyncio.to_thread(
        client.query.get, 
        "ChatHistory", ["role", "content", "_additional { id }"]
    )
    response = response.with_near_vector({
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
        response = await asyncio.to_thread(
            client.query.get, 
            "ChatHistory", ["role", "content"]
        )
        response = response.with_near_vector({
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
    return None


def retrieve_conversation_history(user_query, limit=50):
    """Retrieve the most relevant past messages from Weaviate using vector search."""
    query_embedding = generate_embedding(user_query)

    response = client.query.get(
        "ChatHistory", ["role", "content"]
    ).with_near_vector({
        "vector": query_embedding,
        "certainty": 0.7  # Adjust threshold based on testing
    }).with_limit(limit).do()

    if response and "data" in response and "Get" in response["data"] and "ChatHistory" in response["data"]["Get"]:
        return response["data"]["Get"]["ChatHistory"]

    return []


# def store_qa_in_weaviate(qa_conversation):
#     """Stores Q&A dialogue in Weaviate."""
#     try:
#         qa_pairs = json.loads(qa_conversation)  # Convert JSON string to list of dicts
        
#         for entry in qa_pairs:
#             role = entry["role"]
#             content = entry["content"]
#             timestamp = format_rfc3339(datetime.utcnow())

#             embedding = generate_embedding(content) if role == "user" else None

#             # Store in Weaviate
#             client.data_object.create(
#                 data_object={
#                     "role": role,
#                     "content": content,
#                     "timestamp": timestamp,
#                     **({"embedding": embedding} if embedding else {})
#                 },
#                 class_name="ChatHistory"
#             )
        
#         print(f"✅ Stored {len(qa_pairs)} Q&A entries in Weaviate.")
    
#     except Exception as e:
#         print(f"Error storing Q&A in Weaviate: {e}")


def store_qa_in_weaviate(qa_conversation):
    """Stores Q&A dialogue in Weaviate with enhanced error handling."""
    try:
        # Check if qa_conversation is None or empty
        if not qa_conversation:
            print("❌ Error: qa_conversation is None or empty.")
            return

        # Check if qa_conversation is a string and print a snippet for debugging
        if isinstance(qa_conversation, str):
            print(f"📄 Received JSON string (snippet): {qa_conversation[:200]}...")
            qa_pairs = json.loads(qa_conversation)  # Convert JSON string to list of dicts
        elif isinstance(qa_conversation, list):
            print("📄 Received a list of Q&A entries.")
            qa_pairs = qa_conversation  # Already a list of dicts
        else:
            print(f"❌ Invalid input type: {type(qa_conversation)}. Expected str or list.")
            return

        # Check if qa_pairs is a list and has valid entries
        if not isinstance(qa_pairs, list) or not qa_pairs:
            print("❌ No valid Q&A data to store.")
            return

        for entry in qa_pairs:
            role = entry.get("role")
            content = entry.get("content")
            timestamp = datetime.utcnow().isoformat()

            # Check for missing required fields
            if not role or not content:
                print(f"❌ Missing required fields in entry: {entry}")
                continue

            # Store each Q&A entry in Weaviate
            try:
                client.data_object.create(
                    data_object={
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    },
                    class_name="ChatHistory"
                )
            except weaviate.exceptions.UnexpectedStatusCodeException as e:
                print(f"❌ Error storing entry in Weaviate: {e}")
            except Exception as e:
                print(f"❌ Unexpected error storing entry: {e}")

        print(f"✅ Stored {len(qa_pairs)} Q&A entries in Weaviate.")

    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        print(f"❌ Raw JSON content: {qa_conversation[:500]}")  # Print first 500 chars for debugging
    except Exception as e:
        print(f"❌ Error storing Q&A in Weaviate: {e}")




def export_weaviate_data(class_name="ChatHistory"):
    """Exports data from Weaviate and saves it to a JSON file."""
    query = f"""
    {{
      Get {{
        {class_name} {{
          role
          content
          timestamp
          embedding  # Export embeddings if available
        }}
      }}
    }}
    """

    try:
        # Run the query to get data from Weaviate
        response = client.query.raw(query)
        
        # Print the full response if 'data' key is missing
        if 'data' not in response:
            print(f"❌ No 'data' key in response. Full response: {response}")
            return

        data = response['data']['Get'].get(class_name, [])
        
        if not data:
            print(f"❌ No data found for class '{class_name}'. Full response: {response}")
            return
        
        # Save the data to a JSON file
        with open('weaviate_export.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print("\n✅ Data successfully exported to 'weaviate_export.json'")
    
    except weaviate.exceptions.WeaviateException as e:
        print(f"❌ Weaviate error during export: {e}")
    except Exception as e:
        print(f"❌ Unexpected error during export: {e}")


#create_schema()  # Ensure schema is created at startup
#store_chat_history() # Store chat.json chats

#export_weaviate_data()

