# NikBot

NikBot is a conversational bot that simulates a business advisor named "Nik." It uses OpenAI's GPT API, and Embedding to store and generate responses and maintains a persistent conversation history in a `chat.json` file.

## Features

- **Simulates Nik:** Nik is a B2B business advisor who provides strategic advice to users.
- **Persistent Conversations:** Tracks and saves all unique user-bot and user-consultant interactions in `chat.json`.
- **Custom Responses:** Generates tailored responses based on the context of the conversation or from similar previous chats.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key
- A Discord bot token
- Required Python packages (see below)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd NickBot
nano .env file and write your BOT_TOKEN and OPENAI_TOKEN 
python bot.py
