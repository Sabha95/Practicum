from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import requests
import logging
import uuid
import os

from jsonpickle import json

from utils.user_data import (
    get_user_data, set_user_data, log_emotion, get_emotion_history,
    predict_emotion, generate_anonymous_nickname, log_feedback, load_data
)
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)

RASA_API_URL = 'http://localhost:5005/webhooks/rest/webhook'

DATA_FILE = Path(__file__).parent / "data/user_logs.json"

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

anonymous_counter = 0
nickname = ""


@app.before_request
def track_user():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())


@app.route('/consent', methods=["GET", "POST"])
def consent():
    user_id = session.get('user_id')

    # Initialize user session if not exists
    if not user_id:
        session['user_id'] = str(uuid.uuid4())
        user_id = session['user_id']

    user_data = get_user_data(user_id)

    if request.method == "POST":
        # Handle consent and nickname submission
        if "consent" in request.form:
            consent = request.form.get("consent") == "yes"

            if consent:
                nickname: str = request.form.get("nickname", "").strip()
                if nickname:
                    set_user_data(user_id, nickname, consent)
                    user_data = get_user_data(nickname)
                    return render_template('index.html', nickname=nickname, consented=user_data.get("consent"))
                else:
                    return render_template("consent_form.html", error="Please enter a nickname to proceed.")
            else:
                nickname = generate_anonymous_nickname()
                # nickname = f"Anonymous {anonymous_counter}"
                set_user_data(user_id, nickname, consent)
                user_data = get_user_data(user_id)
                return render_template('index.html', nickname=nickname, consented=user_data.get("consent"))


@app.route('/')
def index():
    return render_template('consent_form.html')


@app.route('/webhook', methods=['POST'])
def webhook():
    logging.debug("Entered webhook")
    user_message = request.json['message']
    logging.debug(request.json['message'])
    logging.debug(user_message)

    try:
        rasa_response = requests.post(RASA_API_URL, json={'message': user_message})
        rasa_response_json = rasa_response.json()

        if rasa_response_json:
            bot_response = rasa_response_json[0]['text']
            saved_user_data = get_user_data(request.json['nickname'])
            if saved_user_data and bool(saved_user_data.get('consent')):
                emotion = predict_emotion(user_message)
                log_emotion(request.json['nickname'], user_message, emotion)

        else:
            bot_response = "Sorry, i did not understand that. "

        print({bot_response})
        return jsonify({'response': bot_response})

    except Exception as e:
        print(f"Error is : {e}")


@app.route('/feedback', methods=['POST'])
def feedback():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "No user session"}), 400

    data = request.get_json()
    user_message = data.get("user_message")
    bot_response = data.get("bot_response")
    thumbs_up = data.get("thumbs_up")
    nickname = data.get("nickname")

    if not all([user_message, bot_response]):
        return jsonify({"error": "Missing fields"}), 400

    log_feedback(nickname, bot_response, thumbs_up, user_message)

    # user_data = get_user_data(nickname)
    # feedback_entry = {
    #     "text": user_message,
    #     "bot_response": bot_response,
    #     "thumbs_up": bool(thumbs_up),
    #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    # }
    #
    # if nickname in user_data:
    #     user_data[nickname].setdefault("feedback", []).append(feedback_entry)
    #     set_user_data(user_data)

    return jsonify({"status": "feedback saved"})


@app.route('/emotion_chart.html')
def serve_chart_page():
    """Serves the emotion chart page."""
    # This route just needs to render the chart template.
    return render_template('emotion_chart.html')


@app.route('/api/chatlogs', methods=['GET', 'POST'])
def get_chat_logs():
    """
    This is the REST API endpoint. It reads the JSON data from the file
    and returns it to the client.
    """
    try:
        data = load_data()
        nickname = request.json['nickname']
        if nickname in data:
            user_data = data[nickname]
            return jsonify(user_data), 200
    except FileNotFoundError:
        return jsonify({"error": "Data file not found"}), 404
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
