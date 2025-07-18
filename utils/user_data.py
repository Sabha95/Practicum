import json
import os
import random
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

DATA_PATH = "data/user_logs.json"
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
if not os.path.exists(DATA_PATH):
    with open(DATA_PATH, "w") as f:
        json.dump({}, f)

emo_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emo_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emo_model.eval()


def load_data():
    with open(DATA_PATH) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_data(data):
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_user_data(nickname):
    data = load_data()
    for nickname, users in data.items():
        if data.get("nickname") == nickname:
            return nickname, data
    return data.get(nickname)


def set_user_data(user_id, nickname, consent):
    data = load_data()

    # If the user (by nickname) already exists, preserve their emotion and feedback history
    if nickname in data:
        existing_emotions = data[nickname].get("emotions", [])
        existing_feedback = data[nickname].get("feedback", [])
    else:
        existing_emotions = []
        existing_feedback = []

    # Update or add user entry using nickname as key
    data[nickname] = {
        "nickname": nickname,  # Redundant but informative
        "consent": consent,
        "emotions": existing_emotions,
        "feedback": existing_feedback
    }

    save_data(data)


def log_feedback(nickname, bot_response, feedback,user_message):
    data = load_data()

    if nickname in data:
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "bot_response": bot_response,
            "feedback": "isPositive" if feedback else "isNegative",
            "reward":  1 if feedback else -1,
            "user_message": user_message
        }
        data[nickname]["feedback"].append(entry)
        save_data(data)


def log_emotion(nickname, text, emotion):
    data = load_data()

    if nickname in data:
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "text": text,
            "emotion": emotion,
            "emotion_feedback": "isPositive"
        }
        data[nickname]["emotions"].append(entry)
        save_data(data)


def get_emotion_history(user_id):
    data = load_data()
    return data.get(user_id, {}).get("emotions", [])


def predict_emotion(text):
    inputs = emo_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = emo_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        label = emo_model.config.id2label[predicted_class]
    return label


def generate_anonymous_nickname():
    data = load_data()
    counter = 1
    while f"Anonymous {counter}" in data:
        counter += 1
    return f"Anonymous {counter}"
