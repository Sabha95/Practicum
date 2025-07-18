import os
from pathlib import Path
from typing import Any, Text, Dict, List

from openai import OpenAI
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# --- 1. CRITICAL SECURITY: Load API Key from Environment ---
# In your terminal, run this command BEFORE starting RASA:
# For Linux/macOS: export OPENAI_API_KEY="your_real_api_key_here"
# For Windows CMD:  set OPENAI_API_KEY="your_real_api_key_here"
# For PowerShell:   $env:OPENAI_API_KEY="your_real_api_key_here"
os.environ["OPEN_API_KEY"] = ""
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running the action server.")

client = OpenAI(api_key=os.environ.get("OPEN_API_KEY"))

# --- 2. DYNAMIC MODEL LOADING LOGIC ---

# Define the model you are currently using as a fallback.
# This ensures your bot keeps working even if the tracking file is missing.
DEFAULT_MODEL_ID = "ft:gpt-3.5-turbo-0125:personal::Bspa6enW"


def get_latest_model_id() -> str:
    """
    Reads the model ID from the last line of the tracking file.
    Falls back to a default model if the file is missing or empty.
    """
    try:
        # Assumes this script is in `actions/` and the models file is in `models/`
        project_root = Path(__file__).parent.parent
        model_file_path = project_root / "models" / "latest_gpt_model.txt"

        if not model_file_path.exists():
            print(f"INFO: Model tracking file not found. Using default model: {DEFAULT_MODEL_ID}")
            return DEFAULT_MODEL_ID

        with open(model_file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]  # Read non-empty lines
            if not lines:
                print(f"INFO: Model tracking file is empty. Using default model: {DEFAULT_MODEL_ID}")
                return DEFAULT_MODEL_ID

            latest_model = lines[-1]
            print(f"INFO: Successfully loaded latest model from file: {latest_model}")
            return latest_model

    except Exception as e:
        print(f"ERROR: Error reading model file: {e}. Using default model: {DEFAULT_MODEL_ID}")
        return DEFAULT_MODEL_ID


# Load the model ID ONCE when the action server starts up for efficiency.
CURRENT_GPT_MODEL_ID = get_latest_model_id()


class ActionGenerateLLMResponse(Action):

    def name(self) -> Text:
        return "action_generate_llm_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text")

        try:
            chat_completion = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": user_message
                }],
                # --- 3. USE THE DYNAMICALLY LOADED MODEL ID ---
                model=CURRENT_GPT_MODEL_ID,
                top_p=0.9
            )

            generated_response = chat_completion.choices[0].message.content.strip()
            dispatcher.utter_message(text=generated_response)

        except Exception as e:
            # Provide a user-friendly error message
            print(f"An OpenAI API error occurred: {str(e)}")
            dispatcher.utter_message(
                text="I'm sorry, I'm having a little trouble connecting right now. Please try again in a moment.")

        return []