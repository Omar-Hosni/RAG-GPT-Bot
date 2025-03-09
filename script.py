import os
import json
from openai.embeddings_utils import get_embedding
import openai
import re

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_TOKEN")

# File to store conversation history
CONVERSATION_FILE = "conversation.json"
EMBEDDING_MODEL = "text-embedding-3-small"  # Define the embedding model, better accurracy: text-embedding-ada-002

def add_embeddings_to_conversation():
    try:
        # Load existing conversation history
        if os.path.exists(CONVERSATION_FILE):
            with open(CONVERSATION_FILE, "r", encoding="utf-8") as file:
                conversation = json.load(file)
        else:
            print("Conversation file not found.")
            return

        # Add embeddings to entries without them
        for entry in conversation:
            # Get embedding for the content
            if entry["role"] == "user" and "embedding" not in entry:
                entry["embedding"] = get_embedding(entry["content"], engine=EMBEDDING_MODEL)
                # print(f"Added embedding for: {entry['content']}")

        # Save updated conversation history if changes were made
        with open(CONVERSATION_FILE, "w", encoding="utf-8") as file:
            json.dump(conversation, file, indent=4, ensure_ascii=False)
            print("Conversation updated with embeddings.")

    except Exception as e:
        print(f"Error while adding embeddings: {e}")

# print
def print_the_conv():
    with open(CONVERSATION_FILE, "r", encoding="utf-8") as file:
        conversation = json.load(file)
    for entry in conversation:
        print(entry["content"])


# delete entry with certain question / response
def delete_entry_with_content(content_to_delete):
    try:
        # Load the conversation history
        with open(CONVERSATION_FILE, "r", encoding="utf-8") as file:
            conversation_history = json.load(file)

        # Filter out entries with the specified content
        updated_conversation = [
            entry for entry in conversation_history if entry["content"] != content_to_delete
        ]

        # Save the updated conversation history back to the file
        with open(CONVERSATION_FILE, "w", encoding="utf-8") as file:
            json.dump(updated_conversation, file, indent=4, ensure_ascii=False)

        print(f"Entry with content '{content_to_delete}' deleted successfully.")

    except FileNotFoundError:
        print(f"Error: {CONVERSATION_FILE} not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in {CONVERSATION_FILE}.")
    except Exception as e:
        print(f"Unexpected error: {e}")



def generate_qa_from_large_transcript(transcript_text, chunk_size=250):
    """Converts a large transcript into a Q&A conversation using GPT, splitting if necessary."""
    try:
        # Split the transcript into manageable chunks
        words = transcript_text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        all_qa_conversations = []

        # Process each chunk and append results
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1}/{len(chunks)}...")

            prompt = f"""
            Convert the following transcript into a conversation between a user and an assistant:

            {chunk}

            The user should ask meaningful questions, and the assistant should provide well-structured responses.

            Return the response in JSON format like:
            [
                {{"role": "user", "content": "User's question"}},
                {{"role": "assistant", "content": "Assistant's response"}},
                ...
            ]
            """

            # Make the API call
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are an expert at generating Q&A from transcripts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=750,
            )

            # Extract the content
            raw_content = response['choices'][0]['message']['content']

            # Remove code block markers if they exist
            cleaned_content = re.sub(r"^```json\n|```$", "", raw_content.strip(), flags=re.MULTILINE)

            # Convert to JSON
            qa_conversation = json.loads(cleaned_content)
            all_qa_conversations.extend(qa_conversation)

        # Save the combined JSON data to a file
        with open('combined_qa_conversation.json', 'w', encoding='utf-8') as f:
            json.dump(all_qa_conversations, f, indent=4)

        print("All chunks processed and saved to combined_qa_conversation.json")
        return all_qa_conversations

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print("Check the format of the response. It might not be valid JSON.")
        return None
    except openai.error.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

# Example usage
large_transcript = """Alex: Thanks for coming in, Jordan. I’ve been struggling to scale my business and could really use your advice.

Jordan: Happy to help, Alex. Let’s start with an overview. What’s your main challenge right now?

Alex: Well, sales have plateaued for the past six months, and our marketing efforts don’t seem to be driving much traffic anymore. I feel like we’re just spinning our wheels.

Jordan: I see. What’s your current marketing strategy like? Are you focusing more on paid ads, social media, or organic reach?

Alex: Mostly paid ads and a bit of social media. We’ve been putting a lot into Facebook and Instagram ads, but the ROI isn’t great.

Jordan: Have you tried any other platforms, or is it mainly those two?

Alex: Just those two. We haven’t explored Google Ads or LinkedIn much.

Jordan: Got it. Diversifying your ad spend could help. Also, refining your audience targeting and ad copy might boost your ROI. Have you done any customer feedback or surveys to understand what’s resonating with them?

Alex: Not really. We’ve mainly been relying on analytics from the ad platforms.

Jordan: That’s a common pitfall. Understanding your customer’s pain points directly can inform both your marketing and product development strategies. I’d suggest running a short survey to gather insights.

Alex: That makes sense. What should I ask in the survey?

Jordan: Focus on their challenges, why they chose your product, and what they feel is missing. For instance:

    What problem were you trying to solve with our product?
    What nearly stopped you from buying?
    What feature would you most like to see added?

Alex: Those are good questions. I can definitely set that up.

Jordan: Great! Now, about scaling—are your current operations optimized to handle more customers if marketing improves?

Alex: Honestly, probably not. We’re already stretched thin with fulfillment and customer support.

Jordan: That’s a red flag. Scaling without operational readiness leads to poor customer experiences. We’ll need to look at automating parts of your fulfillment process and maybe outsourcing customer support temporarily.

Alex: I hadn’t thought about outsourcing. Is it expensive?

Jordan: Not necessarily. It depends on the volume and complexity of support needed. There are flexible plans that can fit your budget. We can look into that together.

Alex: That’d be awesome.

Jordan: Perfect. I’ll put together a plan outlining some options for diversifying your ad spend, gathering customer feedback, and streamlining operations. We can go over it next week if that works for you.

Alex: Sounds good! Thanks for all the insight, Jordan. I feel more confident already.

Jordan: Anytime, Alex. Looking forward to our next meeting.
"""

#generate_qa_from_large_transcript(large_transcript)

from pysentimiento import create_analyzer

emotion_analyzer = create_analyzer(task="emotion", lang="en")

def detect_emotion(user, msg):
    result = emotion_analyzer.predict(str(msg))
    for label in result.probas:
        if label in ["joy", "anger"] and result.probas[label] > 0.65:
            return f"User {user} said {msg} feeling emotion: {label}, with percentage: {result.probas[label] * 100:.2f}%"
        else:
            return None

result = emotion_analyzer.predict(str("i feel so angry"))

for label in result.probas:
    if label in ["joy", "anger"] and result.probas[label] > 0.60:
        print(f"{label} is detected")
