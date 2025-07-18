#!/usr/bin/env python3
import json
import os
from pathlib import Path


def process_new_feedback():
    """
    Processes new feedback from a JSON chat log file and converts it into a
    state-action-reward format, designed to be run from the RASA project root.
    """
    # Define paths relative to the project root directory
    project_root = Path(__file__).parent.parent
    json_file_path = project_root / "data" / "user_logs.json"
    processed_marker_file = project_root / "last_processed_feedback.txt"
    output_rl_file = project_root / "rl_data" / "reward.jsonl"  # Using .jsonl

    # Ensure output directories exist
    os.makedirs(output_rl_file.parent, exist_ok=True)

    # Read the timestamp of the last processed entry
    last_processed_time = ""
    if processed_marker_file.exists():
        with open(processed_marker_file, 'r') as f:
            last_processed_time = f.read().strip()

    try:
        with open(json_file_path, 'r') as f:
            chat_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading chat log file: {e}")
        return

    new_rl_data = []
    latest_timestamp = last_processed_time

    for user, data in chat_data.items():
        if "feedback" in data and isinstance(data["feedback"], list):
            for feedback_item in data["feedback"]:
                # Process only new feedback items
                if feedback_item.get("time", "") > last_processed_time:
                    state = feedback_item.get("user_message")
                    action = feedback_item.get("bot_response")
                    reward = feedback_item.get("reward")

                    # Check for valid data AND ensure the reward is positive
                    if state and action and reward is not None and reward > 0:
                        # Define the system message
                        system_message = "You are a helpful mental health assistant who suggests therapy exercises."

                        # Create the final format required by OpenAI
                        formatted_data = {
                            "messages": [
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": state},
                                {"role": "assistant", "content": action}
                            ]
                        }

                        # Append the correctly formatted data
                        new_rl_data.append(formatted_data)
                    # Update the latest timestamp found in this run
                    if feedback_item.get("time", "") > latest_timestamp:
                        latest_timestamp = feedback_item["time"]

    # Save the new data by appending to the .jsonl file
    if new_rl_data:
        with open(output_rl_file, 'a') as f:
            for item in new_rl_data:
                f.write(json.dumps(item) + "\n")
        print(f"Processed and appended {len(new_rl_data)} new feedback entries.")

        # Update the marker file with the latest timestamp
        with open(processed_marker_file, 'w') as f:
            f.write(latest_timestamp)
    else:
        print("No new feedback to process.")


if __name__ == "__main__":
    process_new_feedback()
