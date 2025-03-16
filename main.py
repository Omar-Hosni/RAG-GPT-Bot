import asyncio
import uvicorn
from fastapi import FastAPI
import os
import random
import time
from typing import Final
from dotenv import load_dotenv
import discord
from discord import Intents, Client, Message
from responses import get_response
from video_processing import learn_video_content
from db import store_conversation_entry, find_most_similar_entry
from util import (
    is_travel_related, is_greeting, greetings,
    is_business_or_social_media_related, is_worth_learning,
    introduce_typos, detect_emotion
)

# Load bot token
load_dotenv()
TOKEN: Final[str] = os.getenv('BOT_TOKEN')

# Bot setup
intents = Intents.default()
intents.message_content = True
client = Client(intents=intents)

# Admin username
ADMIN_USERNAME = "meero0445"

# Business channel ID (Replace with actual ID)
BUSINESS_CHANNEL_ID = 1333611536899379294

# Session management
session_store = {}
SESSION_TIMEOUT = 300  # Session timeout in seconds (5 minutes)


##################### CHATBOT HELPER FUNCTIONS #####################

async def send_heartbeats():
    """Send manual heartbeats to Discord to prevent disconnections."""
    while True:
        await asyncio.sleep(5)
        try:
            await client.ws.ping()
            print("💓 Sent manual heartbeat to Discord.")
        except Exception as e:
            print(f"❌ Failed to send heartbeat: {e}")
            continue


def get_session(user_id):
    """Retrieve or create a session for a user."""
    if user_id in session_store:
        last_active, messages = session_store[user_id]
        if time.time() - last_active < SESSION_TIMEOUT:
            session_store[user_id] = (time.time(), messages)
            return messages
        else:
            del session_store[user_id]  # Session expired

    session_store[user_id] = (time.time(), [])
    return session_store[user_id][1]


def update_session(user_id, role, content):
    """Add message to user session."""
    if user_id in session_store:
        last_active, messages = session_store[user_id]
        messages.append({"role": role, "content": content})
        session_store[user_id] = (time.time(), messages)


def clear_session(user_id):
    """Clear session context."""
    if user_id in session_store:
        del session_store[user_id]


##################### CHATBOT WRAPPER FUNCTIONS #####################

async def send_message(message: Message, user_message: str, username: str) -> None:
    """Handles sending responses to users."""
    if not user_message:
        print('(⚠️ Message was empty or intents are disabled.)')
        return

    session_context = get_session(str(message.author.id))

    is_private = user_message.startswith('?')
    if is_private:
        user_message = user_message[1:]  # Remove '?' for processing

    try:
        if is_travel_related(user_message):
            await message.author.send(f"🛪{message.author.name} asked you about your travel plans saying :{user_message}")
            session_context.append({"role": "user", "content": user_message})
            response = await get_response(user_message, str(message.author.id), session_context)
            await message.channel.send(response)
            return

        if is_greeting(user_message):
            await message.channel.send(random.choice(greetings))
            return
        
        emotion = detect_emotion(message.author.name, user_message)
        if emotion:
            print('EMOTION DETECTED')
            await message.author.send(emotion)
        else:
            print("NO EMOTION DETECTED")


        # Get response from AI
        session_context.append({"role": "user", "content": user_message})
        response = await get_response(user_message, str(message.author.id), session_context)
        session_context.append({"role": "assistant", "content": response})

        update_session(str(message.author.id), "assistant", response)
        
        response_with_typo = introduce_typos(response)
        # Send response (Private or Public)
        if is_private:
            await asyncio.sleep(10)
            await message.author.send(response_with_typo, mention_author=True)
        else:
            await asyncio.sleep(10)
            await message.channel.send(f"{message.author.mention} {response_with_typo}", mention_author=True)

    except Exception as e:
        print(f"⚠️ Error in send_message: {e}")


##################### DISCORD EVENT HANDLERS #####################

@client.event
async def on_ready() -> None:
    """Fetches past business/social media messages from Discord and stores them in Weaviate."""
    print(f'🚀 {client.user} is now running & scanning past messages.')

    channel = client.get_channel(BUSINESS_CHANNEL_ID)
    if not channel:
        print("❌ Error: Business channel not found.")
        return

    stored_messages = set()
    new_messages = []

    # scan through history messages, store only once that is both business related and has no similarity in DB
    async for message in channel.history(limit=5000):
        content = message.content.strip()
        if not content or not is_business_or_social_media_related(content):
            continue

        if await find_most_similar_entry(content) is None:
            role = "assistant" if message.author.name == ADMIN_USERNAME else "user"
            new_messages.append({"role": role, "content": content})
            stored_messages.add(content)

    # add the new business-related messages into the DB
    # if new_messages:
    #     for msg in new_messages:
    #         store_conversation_entry(msg["role"], msg["content"], "imported_message")
    #     print(f"✅ Stored {len(new_messages)} new business-related messages in Weaviate.")
    # else:
    #     print("✅ No new business messages to store.")


@client.event
async def on_message(message: Message) -> None:
    """Handles incoming messages and stores business/social media-related Q&A in Weaviate."""
    if message.author == client.user or not message.content.strip():
        return

    user_id = str(message.author.id)
    username = str(message.author)
    user_message = message.content.strip()
    channel_name = str(message.channel)

    print(f'📩 [{channel_name}] {username}: {user_message}')

    # General Channel - Respond with Context
    if channel_name == "general":
        await send_message(message, user_message, username)
        return

    # Business Channels - Store Q/A Conversations
    await handle_business_conversations(username, user_message)

    # Admin Request for Video Learning
    if username == ADMIN_USERNAME and user_message.lower().startswith("learn the content of this video:"):
        await handle_video_learning_request(message, user_message)


async def handle_business_conversations(username: str, user_message: str) -> None:
    """Stores business-related conversations in Weaviate."""
    if not is_business_or_social_media_related(user_message):
        return

    similar_message = await find_most_similar_entry(user_message)

    if not similar_message and is_worth_learning(user_message):
        #store_conversation_entry("user", user_message, "live_chat")
        print("✅ Stored business-related user message.")

    if username == ADMIN_USERNAME:
        #store_conversation_entry("assistant", user_message, "live_chat")
        print("✅ Stored admin response as assistant knowledge.")


async def handle_video_learning_request(message: Message, user_message: str) -> None:
    """Handles admin requests to learn content from YouTube videos."""
    video_url = user_message.split("learn the content of this video:")[-1].strip()

    if "youtube.com" in video_url or "youtu.be" in video_url:
        await message.channel.send("📥 Learning from video, please wait...")
        try:
            await asyncio.to_thread(learn_video_content, video_url)
            await message.channel.send("✅ Video content learned and stored in Weaviate!")
        except Exception as e:
            print(f"❌ Error learning video content: {e}")
            await message.channel.send("❌ Failed to learn video content.")
    else:
        await message.channel.send("❌ Invalid YouTube link.")



##################### MAIN EVENT LOOP #####################

async def run_bot():
    """Runs the Discord bot in an async event loop."""
    try:
        await client.start(TOKEN)  # Use start() instead of run() for async compatibility
        print("bot is running...")
    except discord.errors.ConnectionClosed as e:
        print(f"❌ WebSocket closed, reconnecting in 5 seconds: {e}")
        asyncio.sleep(5)
        await run_bot()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(run_bot())    
