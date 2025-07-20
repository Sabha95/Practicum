import os
from pathlib import Path
from typing import Any, Text, Dict, List

from openai import OpenAI
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from dotenv import load_dotenv
import logging

load_dotenv()

client = OpenAI(api_key=os.environ.get("MY_SECRET_API_KEY"))

DEFAULT_MODEL_ID = "ft:gpt-3.5-turbo-0125:personal::Bspa6enW"

logging.basicConfig(level=logging.DEBUG)


def get_latest_model_id() -> str:
    try:
        project_root = Path(__file__).parent.parent
        model_file_path = project_root / "models" / "latest_gpt_model.txt"

        if not model_file_path.exists():
            print(f"INFO: Model tracking file not found. Using default model: {DEFAULT_MODEL_ID}")
            return DEFAULT_MODEL_ID

        with open(model_file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if not lines:
                print(f"INFO: Model tracking file is empty. Using default model: {DEFAULT_MODEL_ID}")
                return DEFAULT_MODEL_ID

            latest_model = lines[-1]
            print(f"INFO: Successfully loaded latest model from file: {latest_model}")
            return latest_model

    except Exception as e:
        print(f"ERROR: Error reading model file: {e}. Using default model: {DEFAULT_MODEL_ID}")
        return DEFAULT_MODEL_ID


CURRENT_GPT_MODEL_ID = get_latest_model_id()

# 🧠 Define response styles for XAI
STYLE_PROMPTS = {
    "empathic": "You are a caring therapist. Respond empathetically to: ",
    "encouraging": "You are a supportive coach. Respond with encouragement to: ",
    "neutral": "You are a calm assistant. Respond factually to: ",
    "problem_solving": "You are a practical guide. Offer suggestions to: "
}


def choose_response_style(user_message: str, emotion: str, intent: str, intent_confidence: float) -> (str, str):
    """
    Enhanced rule-based logic to pick a style and provide a richer explanation.
    Includes intent and confidence in the explanation.
    """
    emotion = (emotion or "").lower()
    msg = user_message.lower()

    explanation_parts = []  # Accumulate reasons for the style choice

    if any(word in msg for word in ["exam", "fail", "grades", "deadline"]) and emotion in ["sadness", "fear", "anger"]:
        style = "empathic"
        explanation_parts.append(f"Detected emotion '{emotion}' and academic stress keywords.")
    elif "thank" in msg or emotion == "joy":
        style = "encouraging"
        explanation_parts.append(f"User showed appreciation or joy.")
    elif emotion in ["neutral", None]:
        style = "neutral"
        explanation_parts.append(f"No strong emotion detected.")
    else:
        style = "problem_solving"
        explanation_parts.append(f"Emotion '{emotion}' and request context suggest a practical approach.")

    # Add intent information to the explanation
    explanation_parts.append(f"Top intent was '{intent}' with confidence {intent_confidence:.2f}.")

    # Combine the explanation parts
    explanation = " ".join(explanation_parts)
    explanation = "Based on: " + explanation + " Therefore, chose " + style + " style."

    return style, explanation


class ActionGenerateLLMResponse(Action):

    def name(self) -> Text:
        return "action_generate_llm_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        prompt = ""
        logging.debug("entered action.py")
        user_message = tracker.latest_message.get("text")
        intent_ranking = tracker.latest_message.get('intent_ranking', [])
        detected_emotion = tracker.get_slot("emotion")

        if intent_ranking:
            # Get the top guess from the NLU model
            top_intent_guess = intent_ranking[0]['name']
            top_intent_confidence = intent_ranking[0]['confidence']

            prompt += f"My own NLU model is not confident, but its best guess is the intent '{top_intent_guess}' with a confidence of {top_intent_confidence:.2f}. "
            prompt += "Based on this, please provide a helpful and relevant response to the user."
        else:
            # Fallback for the fallback - if NLU data is missing for some reason
            prompt += "I could not classify this input. Please provide a helpful, general response."
            top_intent_guess = "unknown"  # Provide a default value
            top_intent_confidence = 0.0

        # 🔍 Choose response strategy and get explanation
        style, explanation = choose_response_style(user_message, detected_emotion, top_intent_guess,
                                                   top_intent_confidence)
        styled_prompt = STYLE_PROMPTS[style] + user_message

        try:
            chat_completion = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": styled_prompt
                }],
                model=CURRENT_GPT_MODEL_ID,
                top_p=0.9
            )

            generated_response = chat_completion.choices[0].message.content.strip()
            logging.debug("uttring-response ", generated_response)
            # ✅ Respond with main reply
            dispatcher.utter_message(text=generated_response)
            logging.debug("explanation", explanation)
            # ✅ Show or log the explanation (you can remove this from UI in production)
            dispatcher.utter_message(text=f"_Why this response?_ {explanation}")

        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            dispatcher.utter_message(
                text="I'm sorry, I had trouble processing that. Please try again shortly.")

        return []
