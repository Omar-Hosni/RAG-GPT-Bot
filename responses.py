from db import store_conversation_entry, find_most_similar_entry
from gpt import generate_nick_response

async def get_response(user_message, user_id, session_context) -> str:
    """Fetch stored responses or generate a new one using GPT with context."""

    # 1️⃣ Check Weaviate for a similar past query
    similar_response = await find_most_similar_entry(user_message)

    if similar_response:
        print("🔍 Found a Similar Query (Using Stored Response)")
        return similar_response  # ✅ Return stored response

    # 2️⃣ If no match, generate response using GPT with context
    print("🆕 No Similar Query Found (Generating GPT Response)...")
    response = generate_nick_response(user_message, user_id, session_context)

    # 3️⃣ Store new user query & assistant response in Weaviate
    # store_conversation_entry("user", user_message, "generated_user")
    # store_conversation_entry("assistant", response, "generated_assistant")
    return response
