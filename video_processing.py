import os
import re 
import json
import time
import openai
import pytube
from pytube import YouTube, extract
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from dotenv import load_dotenv
from db import store_qa_in_weaviate
import requests

pytube.request.default_range_size = 1048576

load_dotenv()
openai.api_key = os.getenv("OPENAI_TOKEN")


def transcribe_youtube_video(video_url):
    """Fetches captions (subtitles) from a YouTube video."""
    try:
        yt = YouTube(video_url)
        try:
            video_title = yt.title
        except:
            # Fallback if title fetching fails
            video_id = extract.video_id(video_url)
            video_title = f"Video_{video_id}"

        # Extract video ID
        video_id = extract.video_id(video_url)

        # Fetch transcript with error handling
        try:
            captions = YouTubeTranscriptApi.get_transcript(video_id)
        except NoTranscriptFound:
            print("❌ No captions available for this video.")
            return None
        except Exception as e:
            print(f"❌ Error fetching captions: {e}")
            return None

        # Combine captions into a single transcript text
        transcript_text = " ".join([caption['text'] for caption in captions])

        # Properly close video stream
        yt.streams.get_highest_resolution().stream.close()

        return {
            "title": video_title,
            "transcript": transcript_text
        }
    
    except Exception as e:
        print(f"❌ Error fetching video info: {e}")
        return None


def generate_qa_from_transcript(transcript_text):
    """Converts a transcript into a Q&A conversation using GPT."""
    try:
        prompt = f"""
        Convert the following transcript into a conversation between a user and an assistant:
        
        {transcript_text}

        The user should ask meaningful questions, and the assistant should provide well-structured responses.
        
        Return the response in JSON format like:
        [
            {{"role": "user", "content": "User's question"}},
            {{"role": "assistant", "content": "Assistant's response"}},
            ...
        ]
        """

        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "system", "content": "You are an expert at generating Q&A from transcripts."},
                      {"role": "user", "content": prompt}],
            max_tokens=750
        )

        # Extract the content
        raw_content = response['choices'][0]['message']['content']

        # Remove code block markers if they exist
        cleaned_content = re.sub(r"^```json\n|```$", "", raw_content.strip(), flags=re.MULTILINE)

        # Convert to JSON
        qa_conversation = json.loads(cleaned_content)

        return qa_conversation
    
    except Exception as e:
        print(f"Error generating Q&A: {e}")
        return None



def get_transcript_with_headers(video_url):
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}", headers=headers)
        if response.status_code == 403:
            print("❌ Access forbidden. Video might be restricted.")
            return None

        # Fetch captions
        captions = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([caption['text'] for caption in captions])
        
        # Attempt to get video title
        try:
            yt = YouTube(video_url)
            video_title = yt.title
        except Exception as e:
            print(f"⚠️ Failed to get video title, defaulting to 'Untitled Video': {e}")
            video_title = "Untitled Video"  # Default title if error occurs

        # Return both the transcript text and the video title
        return {"text": transcript_text, "title": video_title}

    except Exception as e:
        print(f"❌ Error fetching video info: {e}")
        return None


def generate_qa_from_large_transcript(transcript_text, chunk_size=750):
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


def learn_video_content(video_url):
    """Processes a YouTube video and learns its content."""
    print("📥 Fetching video transcript...")
    transcript_data = get_transcript_with_headers(video_url)
    
    if not transcript_data:
        print("❌ Could not retrieve transcript.")
        return

    print("📝 Generating Q&A from transcript...")
    qa_conversation = generate_qa_from_large_transcript(transcript_data["text"], chunk_size=750)

    if not qa_conversation:
        print("❌ Failed to generate Q&A.")
        return

    print("💾 Storing Q&A in Weaviate...")
    import json
    with open("qa_conversation.json", "w", encoding="utf-8") as f:
        json.dump( qa_conversation, f, ensure_ascii=False, indent=4)

    store_qa_in_weaviate(qa_conversation)
    
    print(f"✅ Finished learning content from video: {transcript_data['title']}")

